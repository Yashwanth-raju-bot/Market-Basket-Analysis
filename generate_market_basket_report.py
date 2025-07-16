print("Script started...")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------ Utility Functions ------------------

def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")

def save_plot(df, kind, title, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 6))
    if kind == 'bar':
        sns.barplot(x=kwargs["x"], y=kwargs["y"], data=kwargs["data"],
                    palette="pastel", ax=ax)
        ax.set_ylabel("")
    elif kind == 'hist':
        sns.histplot(df['Hour'], bins=24, color='skyblue', ax=ax)
    elif kind == 'scatter':
        sns.scatterplot(data=kwargs["data"], x='lift', y='confidence',
                        size='support', hue='support', sizes=(40, 200), ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return plot_to_base64(fig)

# ------------------ Main Analysis ------------------

def run_analysis_generate_report(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(inplace=True)

    # Fix column names to match your CSV
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    # Basket creation
    basket = df.groupby(['BillNo', 'Itemname'])['Quantity'].sum().unstack().fillna(0)
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apriori
    from mlxtend.frequent_patterns import apriori, association_rules
    itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)


    rules = association_rules(itemsets, metric='confidence', min_threshold=0.8)

    # Prune redundant rules
    def is_redundant(rule, rule_set):
        for r in rule_set:
            if (set(rule['antecedents']).issubset(set(r['antecedents'])) and
                set(rule['consequents']).issubset(set(r['consequents'])) and
                not rule.equals(r)):
                return True
        return False

    unique = []
    for _, rule in rules.iterrows():
        if not is_redundant(rule, unique):
            unique.append(rule)

    pruned_rules = pd.DataFrame(unique)
    if pruned_rules.empty:
        print("⚠️ No pruned rules after redundancy removal. Using all rules.")
        pruned_rules = rules.copy()

    # Add rule string for plotting
    pruned_rules['rule'] = pruned_rules.apply(lambda row: f"{set(row['antecedents'])} => {set(row['consequents'])}", axis=1)

    # Save CSVs
    rules.to_csv("association_rules.csv", index=False)
    pruned_rules.to_csv("pruned_association_rules.csv", index=False)

    # Top 20 items
    top_items = df.groupby("Itemname")["Quantity"].sum().sort_values(ascending=False)[:20].reset_index()
    top_item_plot = save_plot(top_items, kind='bar', title="Top 20 Items",
                              x='Quantity', y='Itemname', data=top_items)

    # Top 10 rules
    top10_rules = pruned_rules.sort_values(by='confidence', ascending=False).head(10).reset_index(drop=True)
    top10_plot = save_plot(top10_rules, kind='bar', title="Top 10 Rules by Confidence",
                           x='confidence', y='rule', data=top10_rules)

    scatter_plot = save_plot(pruned_rules, kind='scatter', title="Lift vs Confidence",
                             data=pruned_rules)

    # Convert for HTML
    top10_table = top10_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
    top10_table['antecedents'] = top10_table['antecedents'].astype(str)
    top10_table['consequents'] = top10_table['consequents'].astype(str)

    # Render HTML template
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("report_template.html")
    html_out = template.render(
        top_item_plot=top_item_plot,
        hourly_plot="",  # Skipped
        top10_plot=top10_plot,
        scatter_plot=scatter_plot,
        rules_table=top10_table.to_dict(orient='records')
    )

    with open("market_basket_report.html", "w", encoding='utf-8') as f:
        f.write(html_out)
    print("✅ HTML report generated: market_basket_report.html")


if __name__ == "__main__":
    import sys
    run_analysis_generate_report("market_basket_dataset.csv")
