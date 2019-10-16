from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE


network = MLPRegressor(hidden_layer_sizes=(60, 50), activation='logistic', early_stopping=False)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
lasso = make_pipeline(preprocessing.RobustScaler(), Lasso(alpha=0.0005, random_state=1))


# score = np.sqrt(np.sum(np.square(np.array(network.predict(x2))-np.array(y2)))/(1460*0.3))


class AveragingModel:
    def __init__(self, models):
        self.models = models

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)
        return self

    def predict(self, x1):
        results = np.zeros(x1.iloc[:, 0].shape)
        for model in self.models:
            results = results + model.predict(x1)
        return results / len(self.models)

    def score(self, x2, y2):
        return np.sqrt(np.sum(np.square(np.array(self.predict(x2)) - np.array(y2))) / (1460 * 0.3))


class StackingModel:
    def __init__(self, basemodel, metamodel, n):
        self.metamodel = metamodel
        self.basemodel = basemodel
        self.n = n

    def fit(self, x, y):
        kfold = KFold(n_splits=self.n, shuffle=True, random_state=156)
        self.hold_on_set = pd.DataFrame({'newx': 0, 'newy': 0}, index=[0])
        self.base_models_ = [list() for i in range(self.n)]
        n = 0
        for training_set, hold_on in kfold.split(x):
            base = AveragingModel(self.basemodel)
            # print(type(y.iloc[training_set]))
            base.fit(x.iloc[training_set, :], y.iloc[training_set])
            a = pd.DataFrame({'newx': base.predict(x.iloc[hold_on, :]), 'newy': y.iloc[hold_on]})
            self.hold_on_set = pd.concat([self.hold_on_set, a], axis=0)
            self.base_models_[n] = base
            n += 1
        self.hold_on_set.reset_index(inplace=True, drop=True)
        # print(self.hold_on_set)
        self.metamodel.fit(self.hold_on_set.iloc[:, 0].to_frame(), self.hold_on_set.iloc[:, -1].to_frame())
        return self

    def predict(self, x):
        meta_feature = np.mean([model.predict(x) for model in self.base_models_], axis=0)
        # print(meta_feature)
        return self.metamodel.predict(meta_feature.reshape(-1, 1))

    def score(self, x2, y2):
        return np.sqrt(np.sum(np.square(np.array(self.predict(x2)) - np.array(y2))) / (1460 * 0.4))