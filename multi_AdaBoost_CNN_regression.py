__author__ = 'Xin, Aboozar'  # 作者信息

import numpy as np  # 导入 NumPy 库
from numpy.core.umath_tests import inner1d  # 从 NumPy 中导入内积计算函数
from copy import deepcopy  # 从 copy 模块中导入深拷贝函数

# 导入 Keras 和 CNN 相关模块
from keras.models import Sequential  # 从 keras 导入 Sequential 模型
from sklearn.preprocessing import OneHotEncoder # 从 sklearn 导入 OneHotEncoder 进行标签的独热编码
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from my_custom_layers import MyCustomLayer  # 引入自定义层
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.models import clone_model

class AdaBoostClassifier(object):
    '''
    自定义 AdaBoost 分类器类

    参数
    -----------
    base_estimator: object  # 基础估计器（弱分类器）
    n_estimators: integer, 可选(默认=50)  # 最大估计器（弱分类器）数量
    learning_rate: float, 可选(默认=1)  # 学习率
    algorithm: {'SAMME','SAMME.R'}, 可选(默认='SAMME.R')  # 使用的提升算法类型
    random_state: int or None, 可选(默认=None)  # 随机状态

    属性
    -------------
    estimators_: list of base estimators  # 存储基础估计器的列表
    estimator_weights_: array of floats  # 存储每个基础估计器的权重
    estimator_errors_: array of floats  # 存储每个估计器的分类误差

    输入数据说明：包括特征数据和标签数据。
    1特征数据 (X):
        特征数据通常是一个二维数组（或类似结构），其中每一行代表一个样本，每一列代表一个特征。
        特征应该是数值型的，因为大多数机器学习模型（包括 AdaBoost）都是在数值数据上操作的。
        如果特征数据中包含非数值型数据（如类别型数据），通常需要先将其转换为数值型，例如使用独热编码（One-Hot Encoding）或标签编码（Label Encoding）。
        特征数据的规模（即样本数和特征数）取决于具体的应用场景，但需要足够的样本来让模型能够学习到数据中的模式。
    2标签数据 (y):
        标签数据是一维数组，每个元素对应于特征数据中的一行（一个样本），表示该样本的类别或输出。
        在分类任务中，标签通常是类别型的，可以是字符串或整数。例如，在二分类问题中，标签可能是 [0, 1]；在多分类问题中，标签可以是 [0, 1, 2, ..., n]。
        在使用 AdaBoost 进行训练时，如果标签是字符串，通常需要将其转换为整数或独热编码的形式。
    3数据预处理:
        数据预处理是训练模型之前的重要步骤。它包括标准化或归一化特征数据、处理缺失值、转换类别型特征等。
        在使用基于树的模型（如决策树，经常作为 AdaBoost 的基础估计器）时，特征的标准化或归一化不是必须的，...
        因为树模型不是基于距离的算法。但如果使用的基础估计器是基于距离的模型（如支持向量机、K近邻等），则特征的标准化或归一化变得很重要。

    函数运行说明：
    1初始化 (__init__ 方法):
        首先，当创建 AdaBoost 分类器实例时，__init__ 方法被调用。它初始化分类器的参数，例如基础估计器、学习率、迭代次数等。
    2训练模型 (fit 方法):
        接下来，使用 fit 方法来训练模型。在这个方法中，模型会对训练数据进行多轮迭代。
        在每一轮迭代中，fit 方法会调用 boost 方法来进行提升。
    3提升过程 (boost 方法):
        boost 方法根据选择的算法版本（SAMME 或 SAMME.R）调用 discrete_boost 或 real_boost 方法。
        在这些方法中，将会对基础估计器（例如 CNN）进行训练，并更新样本权重。
        这些方法中也会调用 deepcopy_CNN 来复制一个新的基础估计器实例，以保证独立训练。
    4预测 (predict 和 predict_proba 方法):
        1. SAMME + predict
            在这种情况下，predict方法会综合所有基础估计器的加权投票结果来确定每个样本的类别。每个基础估计器对最终分类的贡献由其在训练过程中的表现决定。
        2. SAMME + predict_proba
            这种情况下，虽然SAMME算法主要基于类别标签，但predict_proba方法仍会计算每个类别的概率。这是通过结合各个基础估计器的加权预测并应用适当的概率计算方法来实现的。
        3. SAMME.R + predict
            在SAMME.R算法下，即使算法在内部使用类别概率进行计算，predict方法最终仍会返回一个确定的类别标签，这是基于加权概率来决定的。
        4. SAMME.R + predict_proba
            在这种情况下，predict_proba方法将利用SAMME.R算法中的概率信息来计算每个类别的概率。由于SAMME.R本身就是基于概率进行计算的，因此这种情况下的概率估计可能会更准确。

    参考资料:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)
    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)
    '''

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0,
                 algorithm='SAMME.R', random_state=None, epochs=10,
                 copy_previous_estimator=True, callbacks=None, batch_size=32): # , *args, **kwargs

        # 初始化类属性
        self.base_estimator_ = base_estimator  # 基础估计器
        self.n_estimators_ = n_estimators  # 弱分类器数量
        self.learning_rate_ = learning_rate  # 学习率
        self.algorithm_ = algorithm  # 使用的算法
        self.random_state_ = random_state  # 随机状态
        self.epochs = epochs  # CNN 训练的迭代次数
        self.copy_previous_estimator = copy_previous_estimator  # 新参数控制是否复制估计器
        self.callbacks = callbacks if isinstance(callbacks, list) else [callbacks]  # 转换为列表
        self.batch_size = batch_size
        self.estimators_ = list()  # 存储训练过的估计器
        self.estimator_weights_ = np.zeros(self.n_estimators_)  # 初始化估计器的权重
        self.estimator_errors_ = np.ones(self.n_estimators_)  # 初始化估计器的错误率

    def _samme_proba(self, estimator, n_classes, X):
        """根据 Zhu 等人的论文中的算法 4，步骤 2，公式 c) 来计算概率。

        参考文献
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        # 使用估计器对输入 X 进行预测，获取每个样本属于每个类别的概率
        proba = estimator.predict(X)

        # 调整概率值以确保后续的对数计算有效。
        # 替换所有小于可表示的最小正浮点数的概率值，避免对数计算中的无效值或负数。
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        # 对概率值取对数
        log_proba = np.log(proba)

        # 按照 SAMME.R 算法调整概率值。这个调整是为了计算每个样本的最终加权概率。
        # 这个步骤是 SAMME.R 算法的核心，它通过调整原始的概率对数来强调错分的样本。
        return (n_classes - 1) * (log_proba - (1. / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.n_samples = X_train.shape[0]

        # 检查标签是否是概率型标签
        if len(y_train.shape) == 2 and y_train.shape[1] > 1:
            # 计算类别数量
            self.n_classes_ = y_train.shape[1]
            self.classes_ = np.arange(self.n_classes_)  # 假设类别为[0, 1, ..., n_classes-1]
        else:
            # 处理非概率型标签
            self.classes_ = np.array(sorted(list(set(y_train))))
            self.n_classes_ = len(self.classes_)

        # 省略原来的独热编码过程，直接使用概率型标签
        # 训练每个估计器
        for iboost in range(self.n_estimators_):
            # 初始化样本权重
            if iboost == 0:
                sample_weights = np.ones(self.n_samples) / self.n_samples
            else:
                sample_weights = self.sample_weights_

            # 训练并更新样本权重
            sample_weights, estimator_weight, estimator_error = self.boost(X_train, y_train, X_val, y_val,
                                                                           sample_weights)

            if estimator_error is None:
                break

            self.sample_weights_ = sample_weights
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error <= 0:
                break

        # 训练结束后调用每个回调的 on_train_end
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end()

        return self

    def boost(self, X_train, y_train, X_val, y_val, sample_weight):
        # 根据选择的 AdaBoost 算法版本执行不同的提升方法
        if self.algorithm_ == 'SAMME':
            # 如果选择的是 SAMME 算法，调用 discrete_boost 方法
            return self.discrete_boost(X_train, y_train, X_val, y_val, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            # 如果选择的是 SAMME.R 算法，调用 real_boost 方法
            return self.real_boost(X_train, y_train, X_val, y_val, sample_weight)

    def real_boost(self, X_train, y_train, X_val, y_val, sample_weight):
        # 初始化或复制模型
        if self.copy_previous_estimator and len(self.estimators_) > 0:
            estimator = self.deepcopy_CNN(self.estimators_[-1])
        else:
            estimator = self.deepcopy_CNN(self.base_estimator_)

        # 如果设置了随机状态，则在估计器中设置这个状态
        if self.random_state_:
            estimator.set_params(random_state=self.random_state_)

        # 训练模型
        if X_val is not None and y_val is not None:
            estimator.fit(X_train, y_train, epochs=self.epochs, sample_weight=sample_weight,
                          batch_size=self.batch_size, validation_data=(X_val, y_val),
                          callbacks=self.callbacks)
        else:
            estimator.fit(X_train, y_train, epochs=self.epochs, sample_weight=sample_weight,
                          batch_size=self.batch_size, callbacks=self.callbacks)

        # 预测训练数据的概率分布
        y_pred = estimator.predict(X_train)

        # 计算样本的错误程度（概率分布与真实分布的偏离）
        # 使用交叉熵作为错误度量：真实分布与预测分布之间的交叉熵
        error_vector = -np.sum(y_train * np.log(y_pred), axis=1)

        # 将错误程度向量结合样本权重，计算加权错误率
        estimator_error = np.average(error_vector, weights=sample_weight)

        # 如果错误率比随机猜测更差，则停止提升
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        estimator_weight = -np.log(estimator_error + 1e-8)

        # 预测训练数据的概率分布
        y_predict_proba = estimator.predict(X_train)

        # 确保概率分布中的最小值不会影响对数计算
        y_predict_proba = np.clip(y_predict_proba, np.finfo(float).eps, 1 - np.finfo(float).eps)

        # 计算交叉熵误差
        cross_entropy_error = -np.sum(y_train * np.log(y_predict_proba), axis=1)
        # 更新样本权重
        sample_weight *= np.exp(self.learning_rate_ * cross_entropy_error)

        # # 确保 y_train 是概率分布形式
        # if y_train.shape[1] != self.n_classes_:
        #     raise ValueError("y_train must be a probability distribution over classes.")
        #
        # # 构造 y_coding，直接使用概率分布标签
        # y_coding = -1. / (self.n_classes_ - 1) * (1 - y_train) + y_train
        #
        # # 计算更新样本权重的中间变量
        # intermediate_variable = (-1. * self.learning_rate_ * (
        #         ((self.n_classes_ - 1) / self.n_classes_) * inner1d(y_coding, np.log(y_predict_proba))))
        #
        # # 更新样本权重
        # sample_weight *= np.exp(intermediate_variable)


        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        sample_weight /= sample_weight_sum

        # 保存当前训练的模型
        self.estimators_.append(estimator)

        return sample_weight, 1, estimator_error

    def deepcopy_CNN(self, base_estimator0):
        estimator = clone_model(base_estimator0)

        estimator.set_weights(base_estimator0.get_weights())
        # 获取基础估计器的优化器配置
        original_optimizer_config = base_estimator0.optimizer.get_config()
        estimator.compile(loss='categorical_crossentropy',
                          optimizer=Adam(**original_optimizer_config),
                          metrics=['categorical_crossentropy'])
        return estimator

    def discrete_boost(self, X_train, y_train, X_val, y_val, sample_weight):
        # 检查是否已经有训练过的估计器
        if self.copy_previous_estimator and len(self.estimators_) > 0:
            # 如果选择复制上一个估计器并且已有训练过的估计器
            estimator = self.deepcopy_CNN(self.estimators_[-1])
        else:
            # 否则，每次都从基础估计器创建一个新的实例
            estimator = self.deepcopy_CNN(self.base_estimator_)

        # 如果设置了随机状态，则在估计器中设置这个状态
        if self.random_state_:
            estimator.set_params(random_state=1)

        # 将标签转换为独热编码格式，适用于 CNN 训练
        lb = OneHotEncoder(sparse_output=False)
        y_train_encoded = y_train.reshape(len(y_train), 1)
        y_train_encoded = lb.fit_transform(y_train_encoded)

        # 训练估计器
        if X_val is not None and y_val is not None:
            estimator.fit(X_train, y_train_encoded, epochs=self.epochs, sample_weight=sample_weight,
                          batch_size=self.batch_size, validation_data=(X_val, y_val),
                          callbacks=self.callbacks)
        else:
            estimator.fit(X_train, y_train_encoded, epochs=self.epochs, sample_weight=sample_weight,
                          batch_size=self.batch_size, callbacks=self.callbacks)

        # 预测训练数据
        y_pred = estimator.predict(X_train)
        # 将 CNN 的输出转换为类别标签
        y_pred_l = np.argmax(y_pred, axis=1)
        # 计算交叉熵误差
        cross_entropy_error = -np.sum(y_train_encoded * np.log(y_pred), axis=1)

        # 计算错误的预测
        incorrect = y_pred_l != y_train
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # 如果错误率为零，直接返回当前状态，结束训练
        if estimator_error == 0:
            return sample_weight, 0, 0, cross_entropy_error.mean()

        # 更新估计器的权重
        estimator_weight = self.learning_rate_ * (
                np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        # 如果估计器的权重小于等于 0，停止提升
        if estimator_weight <= 0:
            return None, None, None

        # 更新样本权重
        sample_weight *= np.exp(estimator_weight * incorrect)

        # 修正样本权重
        sample_weight = np.clip(sample_weight, a_min=np.finfo(float).eps, a_max=None)

        # 计算样本权重的总和
        sample_weight_sum = np.sum(sample_weight, axis=0)

        # 如果样本权重总和小于等于 0，停止提升
        if sample_weight_sum <= 0:
            return None, None, None

        # 归一化样本权重
        sample_weight /= sample_weight_sum

        # 将当前估计器添加到估计器列表中
        self.estimators_.append(estimator)

        # 返回更新后的样本权重，估计器权重和估计器错误率
        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        if self.algorithm_ == 'SAMME.R':
            # 使用SAMME.R，计算所有估计器的加权概率分布
            pred = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)
        else:
            # SAMME算法，选择预测概率最高的类别
            pred = sum((estimator.predict(X) == self.classes_[:, np.newaxis]) * w
                       for estimator, w in zip(self.estimators_, self.estimator_weights_))

        # 返回概率最高的类别
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def predict_proba(self, X):
        if self.algorithm_ == 'SAMME.R':
            proba = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)
        else:
            proba = sum(
                estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_, self.estimator_weights_))

        # 将加权预测概率的总和除以估计器权重的总和以获得平均效果
        proba /= self.estimator_weights_.sum()

        # 指数缩放概率使其在合理范围内
        # 这是 SAMME.R 算法的特性，用于调整预测概率
        proba = np.exp((1. / (self.n_classes_ - 1)) * proba)

        # 归一化预测概率，确保每个样本的类别概率之和为 1
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0  # 避免除以零
        proba /= normalizer

        return proba

