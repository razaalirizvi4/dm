import json
import numpy as np

def fix_lda_cell(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    found = False
    for cell in data['cells']:
        if cell['cell_type'] == 'code' and any('LinearDiscriminantAnalysis' in line for line in (cell['source'] if isinstance(cell['source'], list) else [cell['source']])):
            cell['source'] = [
                "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n",
                "df = original_df.copy()\n",
                "# 1. Handle non-numeric columns first (important for LDA inputs)\n",
                "for col in df.select_dtypes(include=['object']).columns:\n",
                "    df[col] = df[col].astype('category').cat.codes\n",
                "\n",
                "# 2. Separate X and y\n",
                "X = df.drop(columns=[TARGET_COL])\n",
                "y = df[TARGET_COL]\n",
                "\n",
                "# 3. Fill any NaNs that might break LDA\n",
                "X = X.fillna(X.mean())\n",
                "\n",
                "# 4. Apply LDA\n",
                "# For binary classification, n_components must be 1\n",
                "n_classes = len(np.unique(y))\n",
                "lda = LinearDiscriminantAnalysis(n_components=min(X.shape[1], n_classes - 1))\n",
                "X_lda = lda.fit_transform(X, y)\n",
                "\n",
                "lda_df = pd.DataFrame(X_lda, columns=[f'lda{i+1}' for i in range(X_lda.shape[1])])\n",
                "lda_df[TARGET_COL] = y.values\n",
                "metrics = evaluate(lda_df)\n",
                "metrics['technique'] = 'LDA'\n",
                "results.append(metrics)"
            ]
            found = True
            break
            
    if found:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Successfully updated LDA cell.")
    else:
        print("LDA cell not found.")

if __name__ == '__main__':
    fix_lda_cell(r'd:/work/dmProj/Preprocessing_Evaluation.ipynb')
