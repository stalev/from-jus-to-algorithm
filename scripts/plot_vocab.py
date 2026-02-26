#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Проверяем, установлена ли библиотека для умного размещения текста
try:
    from adjustText import adjust_text

    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_plot(input_csv: Path, output_plot: Path, min_count: int = 5,
                  n_low: int = 10, n_high: int = 10):
    logger.info(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Отсекаем редкие слова (шум слева)
    plot_df = df[df['count'] >= min_count].copy()

    if plot_df.empty:
        logger.warning("Not enough data to plot.")
        return

    # Настройка стиля
    sns.set_theme(style="whitegrid")
    # Делаем график большим, чтобы подписям было где разместиться
    plt.figure(figsize=(16, 12))

    # 1. Рисуем ОСНОВНОЕ ОБЛАКО (серые точки)
    sns.scatterplot(
        data=plot_df,
        x='count',
        y='entropy',
        alpha=0.3,
        s=40,
        color='grey',
        edgecolor=None,
        label='Other terms'
    )

    texts = []
    # Сортируем датафрейм по энтропии
    sorted_df = plot_df.sort_values(by='entropy')

    # --- ГРУППА 1: НИЗКАЯ ЭНТРОПИЯ (Якоря / Специфичные термины) ---
    # Берем n_low первых строк
    low_entropy = sorted_df.head(n_low)

    # Рисуем для них большие СИНИЕ точки
    plt.scatter(
        low_entropy['count'], low_entropy['entropy'],
        color='blue', s=100, edgecolors='white', linewidth=1.5,
        label=f'Top {n_low} Low Entropy (Specific)'
    )

    # Добавляем текст в список на отрисовку
    for _, row in low_entropy.iterrows():
        texts.append(plt.text(
            row['count'], row['entropy'], row['lemma'],
            fontsize=12, color='darkblue', weight='bold'
        ))

    # --- ГРУППА 2: ВЫСОКАЯ ЭНТРОПИЯ (Стоп-слова / Общие термины) ---
    # Берем n_high последних строк
    high_entropy = sorted_df.tail(n_high)

    # Рисуем для них большие КРАСНЫЕ точки
    plt.scatter(
        high_entropy['count'], high_entropy['entropy'],
        color='red', s=100, edgecolors='white', linewidth=1.5,
        label=f'Top {n_high} High Entropy (General)'
    )

    # Добавляем текст в список на отрисовку (точно так же, как для нижних)
    for _, row in high_entropy.iterrows():
        texts.append(plt.text(
            row['count'], row['entropy'], row['lemma'],
            fontsize=12, color='darkred', weight='bold'
        ))

    # Оформление осей и заголовка
    plt.xscale('log')
    plt.title(f'Vocabulary Analysis: Frequency vs Entropy (Total: {len(plot_df)})', fontsize=20)
    plt.xlabel('Frequency (Log Scale)', fontsize=14)
    plt.ylabel('Shannon Entropy (Distribution Uniformity)', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.legend(loc='center right', fontsize=10)

    # --- МАГИЯ: РАСТАСКИВАЕМ ПОДПИСИ ---
    if HAS_ADJUST_TEXT and texts:
        logger.info(f"Optimizing positions for {len(texts)} labels...")
        adjust_text(
            texts,
            # Линии-стрелочки от текста к точке
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
            # Насколько агрессивно расталкивать (можно увеличить до 2.0)
            expand_points=(1.5, 1.5),
            force_text=(0.8, 0.8)
        )

    # Сохранение
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {output_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    # Аргументы для количества подписей
    parser.add_argument("--top_low", type=int, default=10)
    parser.add_argument("--top_high", type=int, default=10)

    args = parser.parse_args()

    generate_plot(
        Path(args.input),
        Path(args.output),
        n_low=args.top_low,
        n_high=args.top_high
    )