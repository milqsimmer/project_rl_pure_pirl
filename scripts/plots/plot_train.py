"""Gráficos diagnósticos de treinamento PPO a partir dos CSVs do SB3 Monitor.

Uso (na raiz do projeto):
    python scripts/plots/plot_train.py --run-tag pilot
    python scripts/plots/plot_train.py --pattern 'runs/pure/seed_*/monitor.csv' --window 30

Não confundir o desvio entre episódios com variabilidade entre seeds.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Coluna no monitor.csv, título, nome do arquivo, unidade.
SERIES = [
    ('r', 'Retorno por episódio', 'retorno', 'recompensa'),
    ('is_success', 'Taxa móvel de sucesso', 'sucesso', 'proporção'),
    ('l', 'Duração por episódio', 'duracao', 'decisões'),
    ('distance', 'Distância final', 'distancia_final', 'm'),
    ('torque_abs_integral', 'Esforço articular integrado', 'torque_integrado', 'N·m·s'),
    ('torque_sq_integral', 'Torque quadrático integrado', 'torque_quadratico', '(N·m)²·s'),
    ('energy_abs', 'Energia mecânica absoluta', 'energia', 'J'),
    ('velocity_envelope_rate', 'Passos físicos acima do envelope de velocidade', 'envelope_velocidade', 'fração'),
    ('torque_envelope_rate', 'Passos físicos acima do envelope de torque', 'envelope_torque', 'fração'),
    ('torque_saturation_rate', 'Passos físicos com torque saturado', 'saturacao_torque', 'fração'),
    ('position_envelope_rate', 'Passos físicos fora do envelope articular', 'envelope_articular', 'fração'),
    ('safe_success', 'Sucesso seguro', 'sucesso_seguro', 'proporção'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-tag', default='pilot', help='Tag dos runs, ex.: pilot ou smoke')
    ap.add_argument('--pattern', default=None, help='Glob de monitor.csv (sobrescreve --run-tag)')
    ap.add_argument('--window', type=int, default=30, help='Janela móvel em episódios')
    ap.add_argument('--out-dir', default='figs/train', help='Destino das figuras')
    args = ap.parse_args()
    if args.window < 1:
        ap.error('--window precisa ser >= 1')
    pattern = args.pattern or f'runs/pure/seed_*_{args.run_tag}/monitor.csv'
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        ap.error(f'Nenhum Monitor encontrado para: {pattern}. Rode na raiz do projeto.')
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path, comment='#')
        except (OSError, pd.errors.EmptyDataError) as e:
            print(f'[AVISO] Ignorando {path}: {e}')
            continue
        if df.empty or not {'r', 'l'}.issubset(df.columns):
            print(f'[AVISO] Monitor vazio ou sem r/l: {path}')
            continue
        df['l'] = pd.to_numeric(df['l'], errors='coerce')
        df['global_step'] = df['l'].fillna(0).cumsum()
        df['episode'] = np.arange(1, len(df) + 1)
        frames.append((path.parent.name, df))
        print(f'{path}: {len(df)} episódios, {int(df.global_step.iloc[-1])} passos')
    if not frames:
        ap.error('Nenhum monitor válido encontrado')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, title, slug, unit in SERIES:
        if not any(key in frame.columns for _, frame in frames):
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        drew = False
        for label, df in frames:
            if key not in df:
                continue
            y = pd.to_numeric(df[key], errors='coerce')
            if not y.notna().any():
                continue
            smoothed = y.rolling(args.window, min_periods=1).mean()
            color = ax.plot(df['global_step'], smoothed, lw=2, label=f'{label} (média móvel)')[0].get_color()
            ax.plot(df['global_step'], y, color=color, alpha=0.12, lw=0.65)
            drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set(title=title, xlabel='Passos de treinamento PPO (aprox.)', ylabel=unit)
        if key in ('is_success', 'safe_success', 'velocity_envelope_rate', 'torque_envelope_rate',
                   'torque_saturation_rate', 'position_envelope_rate'):
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        filename = out / f'train_{slug}.png'
        fig.savefig(filename, dpi=165)
        plt.close(fig)
        saved.append(filename)
    print(f'Gerados {len(saved)} gráficos em {out.resolve()}')
    if not saved:
        print('Nenhuma coluna numérica utilizável encontrada no Monitor.')


if __name__ == '__main__':
    main()
