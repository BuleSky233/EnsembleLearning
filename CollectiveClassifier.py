from sklearn import neighbors, svm
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import prettytable as pt


class CollectiveClassifier:
    def __init__(self):
        self.knnmodel = neighbors.KNeighborsClassifier()
        self.svmmodel = svm.SVC()
        self.dtmodel = DecisionTreeClassifier()

    def fit(self, train_X, train_Y):
        self.knnmodel.fit(train_X, train_Y)
        self.svmmodel.fit(train_X, train_Y)
        self.dtmodel.fit(train_X, train_Y)

    def predict(self, test_X):
        knnpredict = self.knnmodel.predict(test_X)
        svmpredict = self.svmmodel.predict(test_X)
        dtpredict = self.dtmodel.predict(test_X)
        collectivepredict = []
        for i in range(len(knnpredict)):
            if knnpredict[i] == dtpredict[i]:
                collectivepredict.append(knnpredict[i])
            else:
                collectivepredict.append(svmpredict[i])
        return collectivepredict

    def score(self, test_X, test_Y):

        knnpredict = self.knnmodel.predict(test_X)
        svmpredict = self.svmmodel.predict(test_X)
        dtpredict = self.dtmodel.predict(test_X)
        print("knn的准确率为",self.knnmodel.score(test_X, test_Y))
        print("svm的准确率为",self.svmmodel.score(test_X, test_Y))
        print("decision tree的准确率为",self.dtmodel.score(test_X, test_Y))

        print("knn各类别的评价指标为")
        print(classification_report(test_Y, knnpredict))
        print("svm各类别的评价指标为")
        print(classification_report(test_Y, svmpredict))
        print("decision tree各类别的评价指标为")
        print(classification_report(test_Y, dtpredict))

        collectivepredict = self.predict(test_X)
        dict = {}
        dict["BC_CML"] = 0
        dict["CP_CML"] = 1
        dict["k562"] = 2
        dict["normal"] = 3
        dict["pre_BC"] = 4
        label_name = ["BC_CML", "CP_CML", "k562", "normal", "pre_BC"]
        label_num = np.zeros(5)

        # 混淆矩阵 5*5
        confusion = np.zeros([5, 5])
        evaluation = np.zeros([5, 4])
        for i in range(len(collectivepredict)):
            confusion[dict[test_Y[i]]][dict[collectivepredict[i]]] += 1
            label_num[dict[test_Y[i]]] += 1
        sum = 0
        for i in confusion[0]:
            sum += i

        result = []
        right = 0

        for i in range(5):
            cur = []
            cur.append(label_name[i])
            colsum = 0
            rowsum = 0
            for j in range(5):
                colsum += confusion[j][i]
            for j in range(5):
                rowsum += confusion[i][j]
            right += confusion[i][i]
            precision = confusion[i][i] / colsum
            recall = confusion[i][i] / rowsum
            f1_score = (2 * precision * recall) / (precision + recall)
            support = label_num[i]
            cur.append(precision)
            cur.append(recall)
            cur.append(f1_score)
            cur.append(support)
            result.append(cur)

        acc=right/len(test_Y)
        print("集成式分类器的准确率为",acc)

        tb = pt.PrettyTable()
        tb.field_names = ["type", "precision", "recall", "f1-score", "support"]
        for i in range(5):
            tb.add_row(result[i])
        print("集成式分类器各类别的评价指标为")
        print(tb)
