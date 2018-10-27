import os
import pandas as pd
import os.path as osp
from sklearn.model_selection import StratifiedKFold
import pickle

class DataGenerate():

    def __init__(self, train_data_path, test_data_path, to_dir):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.to_dir = to_dir

    def train_generate(self):
        label_warp = {'正常': 0,
                      '不导电': 1,
                      '擦花': 2,
                      '横条压凹': 3,
                      '桔皮': 4,
                      '漏底': 5,
                      '碰伤': 6,
                      '起坑': 7,
                      '凸粉': 8,
                      '涂层开裂': 9,
                      '脏点': 10,
                      '其他': 11,
                      }
        img_path, label = [], []
        for first_path in os.listdir(self.train_data_path):
            first_path = osp.join(self.train_data_path, first_path)
            if '无瑕疵样本' in first_path:
                for img in os.listdir(first_path):
                    img_path.append(osp.join(first_path, img))
                    label.append('正常')
            else:
                for second_path in os.listdir(first_path):
                    defect_label = second_path
                    second_path = osp.join(first_path, second_path)
                    if defect_label != '其他':
                        for img in os.listdir(second_path):
                            img_path.append(osp.join(second_path, img))
                            label.append(defect_label)
                    else:
                        for third_path in os.listdir(second_path):
                            third_path = osp.join(second_path, third_path)
                            if osp.isdir(third_path):
                                for img in os.listdir(third_path):
                                    if 'DS_Store' not in img:
                                        img_path.append(osp.join(third_path, img))
                                        label.append(defect_label)

        df_train = pd.DataFrame({'img_path': img_path, 'label': label})
        df_train['label'] = df_train['label'].map(label_warp)
        df_train = df_train.sample(frac=1.0)
        return df_train

    def train_split(self, df_train, n_folds):
        """
        将原数据集通过k折分层采样， 划分出10个(训练集+验证集),并将其存放至指定文件夹
        Args:
            df_train: train_generate函数生成的df
            n_folds: 折数

        Returns:

        """
        X = df_train['img_path'].values
        y = df_train['label'].values

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
        for i, (train_idx, vali_idx) in enumerate(skf.split(X, y)):
            #pdb.set_trace()

            X_train, y_train = X[train_idx], y[train_idx]
            X_vali, y_vali = X[vali_idx], y[vali_idx]
            data = (X_train.tolist(), y_train.tolist(), X_vali.tolist(), y_vali.tolist())

            to_path = self.to_dir + '/' + 'data_' + str(i+1) + '.pkl'
            with open(to_path, 'wb') as f:
                pickle.dump(data, f)

            print("第{}折数据集已经生成！".format(i+1))

    def test_generate(self):
        all_test_img = os.listdir(self.test_data_path)
        test_img_path = []
        for img in all_test_img:
            if osp.splitext(img)[1] == '.jpg':
                test_img_path.append(osp.join(self.test_data_path, img))

        to_path = self.to_dir +  '/data_test.pkl'
        with open(to_path, 'wb') as f:
            pickle.dump(test_img_path, f)
        print("测试集数据已经生成")
        return test_img_path

if __name__ == '__main__':

    train_data_path = 'data/guangdong_round1_train2_20180916'
    test_data_path = 'data/guangdong_round1_test_b_20181009'
    to_dir = 'data/data_processed'

    data_generator = DataGenerate(train_data_path=train_data_path, test_data_path=test_data_path, to_dir=to_dir)
    df_train = data_generator.train_generate()
    data_generator.train_split(df_train=df_train, n_folds=10)
    data_generator.test_generate()


