# %% [markdown]
# # Final Prediction Pipeline: Model Comparison and Evaluation

# %%
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Load Data and Embeddings

# %%
# Load base data and embeddings
df = pl.read_parquet("merged_final.parquet")
y = df["classification"].to_numpy()

# Load all embeddings
url_embeddings = np.load("url_embeddings.npy")
count_embeddings = np.load("count_vectorizer_embeddings.npy")
tfidf_embeddings = np.load("tfidf_vectorizer_embeddings.npy")
neighbor_embeddings = np.load("neighbor_embeddings.npy")

# Combine all features
X_combined = np.hstack([
    url_embeddings,
    count_embeddings,
    tfidf_embeddings,
    neighbor_embeddings
])

print(f"Combined feature shape: {X_combined.shape}")
print(f"Target distribution: {np.unique(y, return_counts=True)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# %% [markdown]
# ## 2. Train-Test Split (80/20, Stratified)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# %% [markdown]
# ## 3. Model Definitions and Pipelines

# %%
# Define models with appropriate preprocessing
models = {
    'SVM (Linear)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear', class_weight='balanced', random_state=42))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
}

# Naive Bayes requires non-negative features - use only count_embeddings
X_train_nb = count_embeddings[train_test_split(np.arange(len(count_embeddings)), test_size=0.2, random_state=42, stratify=y_encoded)[0]]
X_test_nb = count_embeddings[train_test_split(np.arange(len(count_embeddings)), test_size=0.2, random_state=42, stratify=y_encoded)[1]]

models_nb = {
    'Naive Bayes': MultinomialNB()
}

# %% [markdown]
# ## 4. Cross-Validation with Stratified 10-Fold

# %%
def evaluate_model_cv(model, X, y, model_name):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    return f1_scores

# Evaluate all models
cv_results = {}
print("Performing 10-fold Stratified Cross-Validation...")

for name, model in models.items():
    scores = evaluate_model_cv(model, X_train, y_train, name)
    cv_results[name] = scores
    print(f"{name}: F1-macro = {scores.mean():.4f} ± {scores.std():.4f}")

for name, model in models_nb.items():
    scores = evaluate_model_cv(model, X_train_nb, y_train, name)
    cv_results[name] = scores
    print(f"{name}: F1-macro = {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# ## 5. Visualization: Cross-Validation Results

# %%
plt.figure(figsize=(12, 6))
model_names = list(cv_results.keys())
means = [cv_results[name].mean() for name in model_names]
stds = [cv_results[name].std() for name in model_names]

plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1-macro Score')
plt.title('10-Fold CV F1-macro Scores by Model')
plt.tight_layout()
plt.show()

# Box plots for detailed distribution
plt.figure(figsize=(12, 6))
plt.boxplot([cv_results[name] for name in model_names], labels=model_names)
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1-macro Score')
plt.title('Distribution of 10-Fold CV F1-macro Scores')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Final Model Training and Prediction

# %%
# Train all models on full training set
trained_models = {}
predictions = {}

# Train standard models
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    predictions[name] = model.predict(X_test)

# Train Naive Bayes
for name, model in models_nb.items():
    model.fit(X_train_nb, y_train)
    trained_models[name] = model
    predictions[name] = model.predict(X_test_nb)

# %% [markdown]
# ## 7. Test Set Evaluation

# %%
test_results = {}
for name, pred in predictions.items():
    if name == 'Naive Bayes':
        y_test_actual = y_test  # Same y_test since same split
        f1 = f1_score(y_test_actual, pred, average='macro')
    else:
        f1 = f1_score(y_test, pred, average='macro')
    test_results[name] = f1
    print(f"\n{name} - Test F1-macro: {f1:.4f}")
    print(classification_report(y_test_actual if name == 'Naive Bayes' else y_test, 
                              pred, target_names=le.classes_))

# %% [markdown]
# ## 8. Confusion Matrices

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (name, pred) in enumerate(predictions.items()):
    if idx >= 6:  # Only show first 6 models
        break
        
    if name == 'Naive Bayes':
        cm = confusion_matrix(y_test, pred)
    else:
        cm = confusion_matrix(y_test, pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[idx])
    axes[idx].set_title(f'{name}\nF1-macro: {test_results[name]:.4f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

# Hide unused subplots
for idx in range(len(predictions), 6):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Detailed Metrics Comparison

# %%
# Create detailed metrics table
metrics_data = []
for name, pred in predictions.items():
    if name == 'Naive Bayes':
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average=None)
        macro_f1 = f1_score(y_test, pred, average='macro')
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average=None)
        macro_f1 = f1_score(y_test, pred, average='macro')
    
    for i, class_name in enumerate(le.classes_):
        metrics_data.append({
            'Model': name,
            'Class': class_name,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-score': f1[i]
        })
    metrics_data.append({
        'Model': name,
        'Class': 'Macro Average',
        'Precision': precision.mean(),
        'Recall': recall.mean(),
        'F1-score': macro_f1
    })

metrics_df = pl.DataFrame(metrics_data)
print("\nDetailed Metrics by Model and Class:")
print(metrics_df)

# %% [markdown]
# ## 10. Best Model Selection and Final Results

# %%
best_model_name = max(test_results, key=test_results.get)
best_model = trained_models[best_model_name]
best_f1 = test_results[best_model_name]

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"TEST F1-MACRO SCORE: {best_f1:.4f}")
print(f"{'='*60}")

# Final prediction on test set with best model
if best_model_name == 'Naive Bayes':
    final_predictions = best_model.predict(X_test_nb)
else:
    final_predictions = best_model.predict(X_test)

# Convert back to original labels
final_predictions_labels = le.inverse_transform(final_predictions)
y_test_labels = le.inverse_transform(y_test)

# Save final predictions if needed
# np.save('final_predictions.npy', final_predictions_labels)

print("\nFinal Classification Report:")
print(classification_report(y_test_labels, final_predictions_labels))

# %% [markdown]
# ## 11. Model Performance Summary Plot

# %%
# Create summary bar plot
plt.figure(figsize=(10, 6))
model_names_final = list(test_results.keys())
f1_scores_final = list(test_results.values())

colors = ['red' if name == best_model_name else 'skyblue' for name in model_names_final]

plt.bar(model_names_final, f1_scores_final, color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1-macro Score (Test Set)')
plt.title(f'Model Comparison on Test Set\nBest Model: {best_model_name} ({best_f1:.4f})')
plt.tight_layout()
plt.show()