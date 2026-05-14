# kaggle_workplace

# kaggle_workplace

上位コンペ者のコード分析

### 1. ライブラリのインポート
ここでは機械学習に必要な各種モデルやツールを読み込みます。

```python
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
