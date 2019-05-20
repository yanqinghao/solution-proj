#-*- encoding: UTF-8 -*-
from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import df2table
from mlstudiosdk.modules.algo.data import Domain, Table
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
import pandas as pd
import numpy as np
import codecs
import langid
import jieba
import MeCab
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import scipy.cluster.hierarchy as sch
import warnings
warnings.filterwarnings('ignore')


class Customer_fcluster(LComponent):
    category = 'Nature Language Processing'
    name = "Customer fcluster"
    title = "Customer fcluster"

    inputs = [("Data", Table, "set_data")]
    outputs = [("Grouped Data", Table),
               ("Columns", list),
               ("Metas", list),
               ("Metric Score", MetricFrame),
               ("Metric", MetricType)]

    def __init__(self):
        super().__init__()
        self.data = None
        self.desc_col = None
        self.name_col = None

    def set_data(self, data):
        self.data = data

    def sentence_cut(self, sentence, stopwords):
        if type(sentence) == float:
            texts1 = ''
        else:
            if langid.classify(sentence)[0] == 'ja':
                mecab = MeCab.Tagger("-Owakati")
                texts = mecab.parse(sentence)
            else:
                texts = ' '.join(jieba.cut(sentence))
            texts1 = [word for word in texts.split() if word not in stopwords]
        return ' '.join(texts1)

    def tfidf_matrix(self, corpus):
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()

        vec = vectorizer.fit_transform(corpus)
        tfidf = transformer.fit_transform(vec)

        word = vectorizer.get_feature_names()
        print("特征数:", len(word))

        matrix = tfidf.toarray()
        return matrix

    def run_desc(self, raw_data, stopwords):
        # 分词
        raw_data['desc_jb'] = raw_data[self.desc_col].apply(lambda x: self.sentence_cut(x, stopwords))
        corpus = np.array(raw_data['desc_jb']).tolist()
        # tfidf
        tfidf_df = self.tfidf_matrix(corpus)
        # fcluster
        disMat = sch.distance.pdist(tfidf_df, 'cosine')
        disMat[np.where(np.isnan(disMat))] = np.nanmax(disMat)
        Z = sch.linkage(disMat, method='average')
        cluster = sch.fcluster(Z, t=0.9)
        # print('分为{}组'.format(len(np.unique(cluster))))

        raw_data['cluster'] = cluster
        raw_data['cluster'][raw_data[self.desc_col].apply(lambda x: x.strip() == '')] = 0
        #         print('分为{}组'.format(len(np.unique(raw_data['cluster']))))
        return cluster, raw_data

    def run_name(self, df, stopwords):
        # 把同一cluster的name连接，作为新的聚类数据
        df['name_add_blank'] = [s + ' ' for s in df[self.name_col]]
        grouped = df['name_add_blank'].groupby(df['cluster'])
        df_name_clsum = pd.DataFrame(grouped.sum()).reset_index()
        df_name_clsum.columns = ['cluster', 'name_clsum']
        df_v2 = pd.merge(df, df_name_clsum, how='left', on=['cluster'])
        # 分词
        df_v2['name_clsum_jb'] = df_v2['name_clsum'].apply(lambda x: self.sentence_cut(x, stopwords))
        corpus = np.array(df_v2['name_clsum_jb']).tolist()
        # tfidf
        tfidf_df = self.tfidf_matrix(corpus)
        # fcluster
        disMat = sch.distance.pdist(tfidf_df, 'cosine')
        disMat[np.where(np.isnan(disMat))] = np.nanmax(disMat)
        Z = sch.linkage(disMat, method='average')
        cluster = sch.fcluster(Z, t=0.3, criterion='distance')

        # df_v2['cluster_f'] = cluster
        # df_v2['cluster_f'][df_v2['name_clsum'].apply(lambda x: x.strip() == '')] = 0
        #         print('分为{}组'.format(len(np.unique(df_v2['cluster_f']))))
        return cluster

    def merge_data(self, origin, result):
        # result table to origin table's meta
        domain = origin.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        new_domain = Domain(attributes, classes, meta_attrs + result.domain.variables)
        new_table = Table.from_table(new_domain, origin)
        for attr in result.domain:
            new_table.get_column_view(attr)[0][:] = result.get_column_view(attr)[0][:]
        return new_table

    def run(self):
        # 读入data
        raw_data = table2df(self.data)
        self.desc_col = raw_data.columns.tolist()[1]
        self.name_col = raw_data.columns.tolist()[0]
        raw_data[self.name_col][raw_data[self.name_col].isnull()] = ''
        raw_data = raw_data.reset_index(drop=True)
        # 读入stopwords
        stopwords = [line.rstrip() for line in codecs.open(r"stopwords.txt", 'r', 'utf-8-sig')]
        # 按照desc聚类
        raw_cluster, df = self.run_desc(raw_data, stopwords)
        # 再次按照name聚类
        final_cluster = self.run_name(df, stopwords)

        # 输出
        cols = [final_cluster.reshape(-1, 1)]
        metas = [ContinuousVariable('final_cluster')]
        # metas = [ContinuousVariable('raw_cluster'), ContinuousVariable('final_cluster')]
        tbl = final_cluster.reshape(-1, 1)
        # tbl = np.column_stack((raw_cluster, final_cluster)).reshape(-1, 2)
        res = Table.from_numpy(Domain(metas), tbl)
        finaldata = self.merge_data(self.data, res)

        self.send("Grouped Data", finaldata)
        self.send("Columns", cols)
        self.send("Metas", metas)
        self.send("Metric Score", None)
        self.send("Metric", None)

