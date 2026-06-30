from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="ticks")


def regression_plot_with_stats(
    df,
    col1,
    col2,
    x_label,
    y_label,
    title,
    pearsonr=True,
    spearmanr=True,
    equation=True,
    equal_axis=True,
    save_path=None,
    ax=None,
    drop_outlier_quantile=None,
    print_stats=True,
):
    fed_ax = ax is not None
    if ax is None:
        figsize = (6, 6) if equal_axis else (7, 5)
        fig, ax = plt.subplots(figsize=figsize)
    if drop_outlier_quantile is not None:
        df = df.copy()
        for col in [col1, col2]:
            q_low, q_high = df[col1].quantile(
                [1.0 - drop_outlier_quantile, drop_outlier_quantile]
            )
            df = df[(df[col] <= q_high) & (df[col] >= q_low)]
    sns.regplot(
        x=col1,
        y=col2,
        data=df,
        ci=99,
        marker=".",
        scatter_kws={"color": "0.3", "alpha": 0.5, "s": 40},
        line_kws={"color": "red", "linewidth": 2},
        ax=ax,
    )

    annotation_lines = [f"N = {len(df)}"]
    r, p_val = stats.pearsonr(df[col1], df[col2])
    rho, p_val_sp = stats.spearmanr(df[col1], df[col2])
    slope, intercept, _, _, _ = stats.linregress(df[col1], df[col2])
    if pearsonr:
        annotation_lines.append(f"Pearson $r$: {r:.3f} ($p$={p_val:.2e})")

    if spearmanr:
        annotation_lines.append(f"Spearman $\\rho$: {rho:.3f} ($p$={p_val_sp:.2e})")

    if equation:
        sign = "+" if intercept >= 0 else "-"
        annotation_lines.append(
            f"Equation: $y$ = {slope:.2f}$x$ {sign} {abs(intercept):.2f}"
        )

    annotation_text = "\n".join(annotation_lines)
    ax.text(
        0.05,
        0.95,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    if print_stats:
        print(f"Stats for {title}:")
        print("-" * 40)
        print(f"N = {len(df)}")
        print(f"Pearson r: {r:.3f}, p-value: {p_val:.2e}")
        print(f"Spearman rho: {rho:.3f}, p-value: {p_val_sp:.2e}")
        print(f"Linear regression equation: y = {slope:.2f}x + {intercept:.2f}")
        if drop_outlier_quantile is not None:
            print(
                f"Filtered outliers using quantile: {(1.00 - drop_outlier_quantile):.1e}-{drop_outlier_quantile:.1e}"
            )

    if equal_axis:
        # Find global limits to ensure a perfectly matching grid range
        mn = min(df[col1].min(), df[col2].min())
        mx = max(df[col1].max(), df[col2].max())
        # Add a small padding buffer so points near margins aren't clipped
        padding = (mx - mn) * 0.05
        limits = [mn - padding, mx + padding]

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")

        # Identity line spans the whole plot window
        ax.plot(limits, limits, color="gray", linestyle="--", linewidth=1.2, zorder=0)
    else:
        # Fallback identity line spanning x-axis bounds
        # ax.plot(
        #     [df[col1].min(), df[col1].max()],
        #     [df[col1].min(), df[col1].max()],
        #     color="gray",
        #     linestyle="--",
        #     linewidth=1.2,
        #     zorder=0,
        # )
        pass
    ax.set_xlabel(x_label, fontsize=11, labelpad=8)
    ax.set_ylabel(y_label, fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    sns.despine()
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)  # Saved at high resolution

    if not fed_ax:
        print(f"Displaying plot: {title}")
        plt.show()
