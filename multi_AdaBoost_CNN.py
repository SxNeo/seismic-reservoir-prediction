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
        # if kwargs and args:
        #     # 如果同时传递了位置参数和关键字参数，抛出异常
        #     raise ValueError(
        #         '''AdaBoostClassifier can only be called with keyword
        #            arguments for the following keywords: base_estimator ,n_estimators,
        #             learning_rate, algorithm, random_state''')
        #
        # # 允许的关键字参数列表
        # allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state', 'epochs', 'copy_previous_estimator']
        # # 获取传入的关键字参数
        # keywords_used = kwargs.keys()
        # for keyword in keywords_used:
        #     # 检查每个关键字是否在允许的列表中，如果不在则抛出异常
        #     if keyword not in allowed_keys:
        #         raise ValueError(keyword + ":  Wrong keyword used --- check spelling")
        #
        # # 设置默认值
        # n_estimators = 50  # 弱分类器的默认数量
        # learning_rate = 1  # 学习率的默认值
        # algorithm = 'SAMME.R'  # 默认使用的算法
        # random_state = None  # 默认的随机状态
        # # CNN 特定参数
        # epochs = 10  # CNN 训练的迭代次数
        # copy_previous_estimator = True
        # callbacks = None
        #
        # # 如果提供了关键字参数且没有位置参数
        # if kwargs and not args:
        #     # 根据关键字参数设置对应的属性
        #     if 'base_estimator' in kwargs:
        #         base_estimator = kwargs.pop('base_estimator')
        #     else:
        #         # 如果没有提供 base_estimator，则抛出异常
        #         raise ValueError('''base_estimator can not be None''')
        #     if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
        #     if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
        #     if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
        #     if 'copy_previous_estimator' in kwargs: copy_previous_estimator = kwargs.pop('copy_previous_estimator')
        #     if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
        #     # CNN 特定参数
        #     if 'epochs' in kwargs: epochs = kwargs.pop('epochs')

        # 初始化类属性
        self.base_estimator_ = base_estimator  # 基础估计器
        self.n_estimators_ = n_estimators  # 弱分类器数量
        self.learning_rate_ = learning_rate  # 学习率
        self.algorithm_ = algorithm  # 使用的算法
        self.random_state_ = random_state  # 随机状态
        self.epochs = epochs  # CNN 训练的迭代次数
        self.copy_previous_estimator = copy_previous_estimator  # 新参数控制是否复制估计器
        self.callbacks = callbacks if callbacks is not None else [] # 添加默认为空列表的callbacks参数
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
        # 计算输入数据 X 的样本数量
        self.n_samples = X_train.shape[0]
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        # 对类别标签进行排序，确保预测结果的类别顺序一致
        self.classes_ = np.array(sorted(list(set(y_train))))
        # 如果y是一个独热编码的标签数组，其中每一行代表一个样本，每一列中只有一个位置为1（表示该样本的类别）
        # One - Hot Encoding:
        # 1: [1, 0, 0, 0]
        # 2: [0, 1, 0, 0]
        # 3: [0, 0, 1, 0]
        # 4: [0, 0, 0, 1]
        # yl = np.argmax(y)
        # self.classes_ = np.array(sorted(list(set(yl))))
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        # 计算类别的数量
        self.n_classes_ = len(self.classes_)

        # 验证数据转换为独热编码
        # if X_val is not None and y_val is not None:
        #     lb = OneHotEncoder(sparse=False)
        #     y_val_encoded = lb.fit_transform(y_val.reshape(-1, 1))
        # else:
        #     X_val, y_val_encoded = None, None

        # 对每个估计器进行迭代训练
        for iboost in range(self.n_estimators_):
            # 在第一次迭代中初始化样本权重为均等
            if iboost == 0:
                sample_weights = np.ones(self.n_samples) / self.n_samples
                self.sample_weights_ = sample_weights
            else:
                sample_weights = self.sample_weights_
            # 调用 boost 方法来训练当前的估计器并更新样本权重
            sample_weights, estimator_weight, estimator_error = self.boost(X_train, y_train, X_val, y_val, sample_weights)


            # 如果检测到早停条件（比如估计器错误率为 None），则停止训练
            if estimator_error == None:
                break

            # 记录当前估计器的错误率和权重
            self.sample_weights_ = sample_weights
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            # 如果估计器的错误率小于或等于 0，停止迭代
            if estimator_error <= 0:
                break

        # 返回当前对象，以便可以链式调用其他方法
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
        # 检查是否已经有训练过的估计器
        if self.copy_previous_estimator and len(self.estimators_) > 0:
            # 如果选择复制上一个估计器并且已有训练过的估计器
            estimator = self.deepcopy_CNN(self.estimators_[-1])
        else:
            # 否则，每次都从基础估计器创建一个新的实例
            estimator = self.deepcopy_CNN(self.base_estimator_)

        # 如果设置了随机状态，则在估计器中设置这个状态
        if self.random_state_:
            estimator.set_params(random_state=self.random_state_)

        # 将标签转换为独热编码格式，适用于 CNN 训练
        lb = OneHotEncoder(sparse=False)
        y_train_encoded = y_train.reshape(len(y_train), 1)
        y_train_encoded = lb.fit_transform(y_train_encoded)

        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.reshape(len(y_val), 1)
            y_val_encoded = lb.fit_transform(y_val_encoded)

        if X_val is not None and y_val is not None:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, validation_data=(X_val, y_val_encoded),
                          callbacks=self.callbacks)
        else:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, callbacks=self.callbacks)

        # 预测训练数据
        y_pred = estimator.predict(X_train)
        # 将 CNN 的输出转换为类别标签
        y_pred_l = np.argmax(y_pred, axis=1)
        # 计算错误的预测
        incorrect = y_pred_l != y_train

        # 计算估计器的错误率
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # 如果错误率比随机猜测还差，则停止提升
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        # 再次预测获取概率估计
        y_predict_proba = estimator.predict(X_train)# 在 Keras 中，通常使用 predict 方法来获取模型对于每个类别的预测概率
        # 替换概率中的零值，防止计算对数时出现问题
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        # 准备用于更新样本权重的计算
        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y_train[:, np.newaxis])
        # 计算用于更新样本权重的中间变量
        intermediate_variable = (-1. * self.learning_rate_ * (
                    ((self.n_classes_ - 1) / self.n_classes_) * inner1d(y_coding, np.log(y_predict_proba))))

        # 更新样本权重
        sample_weight *= np.exp(intermediate_variable)
        # 计算样本权重的总和
        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # 归一化样本权重
        sample_weight /= sample_weight_sum

        # 将当前估计器添加到列表中
        self.estimators_.append(estimator)

        # 返回更新后的样本权重，估计器权重（这里设为 1），和估计器错误率
        return sample_weight, 1, estimator_error

    def deepcopy_CNN(self, base_estimator0):
        # 从传入的基础估计器（CNN）创建一个深拷贝
        # 获取基础估计器的配置
        config = base_estimator0.get_config()

        # 根据配置创建一个新的 Sequential 模型
        # 这里注释掉的是另一种创建模型的方式，但未被使用
        # estimator = Models.model_from_config(config)
        estimator = Sequential.from_config(config)

        # 获取基础估计器的权重
        weights = base_estimator0.get_weights()
        # 将这些权重设置到新创建的模型中
        estimator.set_weights(weights)

        # 根据 use_initial_optimizer_config 决定使用的学习率配置
        if self.copy_previous_estimator:
            # 使用当前的优化器配置（包括当前学习率）
            original_optimizer_config = base_estimator0.optimizer.get_config()
            current_learning_rate = K.get_value(base_estimator0.optimizer.lr)
            original_optimizer_config['learning_rate'] = current_learning_rate
        else:
            # 使用初始学习率（这里需要您自己设定或从某处获取初始学习率）
            original_optimizer_config = base_estimator0.optimizer.get_config()
            initial_learning_rate = 0.01  # 假设的初始学习率
            original_optimizer_config['learning_rate'] = initial_learning_rate

        # 使用更新后的优化器配置来编译新模型
        estimator.compile(loss='categorical_crossentropy',
                          optimizer=Adam(**original_optimizer_config),
                          metrics=['accuracy'])

        # 返回这个新创建且已编译的模型
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
        lb = OneHotEncoder(sparse=False)
        y_train_encoded = y_train.reshape(len(y_train), 1)
        y_train_encoded = lb.fit_transform(y_train_encoded)

        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.reshape(len(y_val), 1)
            y_val_encoded = lb.fit_transform(y_val_encoded)

        # 训练估计器
        if X_val is not None and y_val is not None:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, validation_data=(X_val, y_val_encoded),
                          callbacks=self.callbacks)
        else:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, callbacks=self.callbacks)

        # 预测训练数据
        y_pred = estimator.predict(X_train)
        # 将 CNN 的输出转换为类别标签
        y_pred_l = np.argmax(y_pred, axis=1)
        # 计算错误的预测
        incorrect = y_pred_l != y_train

        # 计算估计器的错误率
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # 如果错误率为零，直接返回当前状态，结束训练
        if estimator_error == 0:
            return sample_weight, 0, 0

        # 更新估计器的权重
        estimator_weight = self.learning_rate_ * (
                np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        # 如果估计器的权重小于等于 0，停止提升
        if estimator_weight <= 0:
            return None, None, None

        # 更新样本权重
        sample_weight *= np.exp(estimator_weight * incorrect)

        # 修正 sample_weight，确保不会出现无效值或者小于等于零的情况
        sample_weight[sample_weight <= 0] = np.finfo(float).eps

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
        # 获取类别数和类别标签
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        # 初始化预测结果变量
        pred = None

        # 如果使用的是 SAMME.R 算法
        if self.algorithm_ == 'SAMME.R':
            # 对于 SAMME.R，所有估计器的权重都被视为 1
            # 计算所有估计器的加权预测概率之和
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # 如果使用的是 SAMME 算法
            # 计算所有估计器的加权预测之和
            # 这里使用了 CNN 的输出来确定每个样本的预测类别
            pred = sum((estimator.predict(X).argmax(axis=1) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        # 将加权预测结果除以所有估计器权重的总和，得到平均效果
        pred /= self.estimator_weights_.sum()

        # 如果是二分类问题
        if n_classes == 2:
            # 对第一列（代表负类的预测）取反
            pred[:, 0] *= -1
            # 计算预测总和，确定最终预测类别
            pred = pred.sum(axis=1)
            # 根据预测的符号选择类别
            return self.classes_.take(pred > 0, axis=0)

        # 对于多分类问题，选择具有最高平均加权预测概率的类别
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def predict_proba(self, X):
        # 如果使用的是 SAMME.R 算法
        if self.algorithm_ == 'SAMME.R':
            # 对于 SAMME.R，所有估计器的权重被视为 1
            # 计算所有估计器的加权预测概率之和
            proba = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)
        else:  # 如果使用的是 SAMME 算法
            # 计算所有估计器的加权预测概率之和
            # 这里使用每个估计器的 predict_proba 方法来获取预测概率
            proba = sum(
                estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_, self.estimator_weights_))

        # 将加权预测概率除以所有估计器权重的总和，得到平均效果
        proba /= self.estimator_weights_.sum()

        # 将概率指数缩放以使其位于合理范围内
        # 这是 SAMME.R 算法的一个特点，用于调整预测概率
        proba = np.exp((1. / (self.n_classes_ - 1)) * proba)

        # 归一化预测概率，确保每个样本的类别概率之和为 1
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0  # 避免除以零
        proba /= normalizer

        # 返回预测概率
        return proba
