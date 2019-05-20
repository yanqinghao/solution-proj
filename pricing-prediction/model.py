from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.algo.data import Table, Domain
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.exceptions.exception_base import Error
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable

import numpy as np
from numpy import nan as NA
import pandas as pd
from pandas import Series, DataFrame
from pandas.tseries.offsets import Day,MonthEnd
from datetime import datetime
from random import shuffle
from sqlalchemy import create_engine
import pymysql, time, os, re, random, sys
from scipy.stats import mode
import statsmodels.api as sm

import xgboost as xgb
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

pd.options.mode.chained_assignment = None
np.random.seed(1000000)

class Recommender(LComponent):
    category = 'Algorithm'
    name = 'Recommender'
    title = 'Bidding Price Recommender'
    inputs = [("Train Data", Table, "set_traindata"),
             ("Test Data", Table, "set_testdata")]
    outputs = [("News", Table),
               ("Predictions", Table),
               ("Evaluation Results", Results),
               ("Metric", MetricType),
               ("Metric Score", MetricFrame),
               ("Columns", list),
               ("Metas", list)]

    def __init__(self):
        super().__init__()
        self.train_raw = None
        self.test_raw = None
        self.label_name = None
        self.train_data = None
        self.test_data = None
        self.xgbmodel = None
        self.rfmodel = None
        self.features = None
        self.train_col = None

        
    def set_traindata(self, data):
        self.train_raw = data
         
            
    def set_testdata(self, data):
        self.test_raw = data
        
        
    def get_top(self, group, n):
        return group.sort_values(['z'],ascending=False).head(n)
    
    
    def pre_process(self, md, sdate):
        u=['create_date']
        d=lambda s: str(s)[:10]
        for i in range(len(u)):
            md[u[i]]=md[u[i]].map(d)
            md[u[i]]=pd.to_datetime(md[u[i]])
        md['month']=md['create_date'].map(lambda s: s.strftime("%Y-%m"))

        try:
          md['dol'] = md['dol'].str.strip("%").astype(float)/100

          md['margin'][md.margin=='%'] = '9999%'
          md['margin'] = md['margin'].str.strip("%").astype(float)/100
          md['margin'][md.margin==99.99]=np.nan

          # print ("\n0-Raw data:\ndate>=201710 and bid_status isin ['Accepted','Approved','Rejected']: " +str(md.shape)+'\n')
        except:
          pass

        ###########――看是客户与联想的第几次交易，看是否第一次出现―――
        customer=md[md.bid_status=='Accepted'][['account_name','opportunity','bid_reference_number','create_date']].drop_duplicates().sort_values(['account_name','create_date'])
        customer['deal_num']=customer.groupby(['account_name'])['create_date'].rank(method='min')
        customer['deal_num']=customer['deal_num'].fillna(-1)
        customer['quotation_rank']=customer.groupby(['account_name','opportunity'])['create_date'].rank(method='min')

        md=pd.merge(md,customer,on=['opportunity','bid_reference_number','account_name','create_date'],how='left')
        md['deal_new']=1-(md['deal_num']>1)*1  ##是否是联想与客户的第一个报价单quotation，即是否新客户
        md['quotation_new']=1-(md['quotation_rank']>1)*1  ##是否是某一个商机opportunity内的第一个报价单quotation


        ####################―――brand不为空―――――――――――――
        md['opportunity'][md.opportunity=='']='blank'
        md['distribution_channel_name']=md['distribution_channel_name'].fillna('blank')
        md['end_customer_nielson_id']=md['end_customer_nielson_id'].fillna('blank')
        md['brand']=md['brand'].fillna('blank')
        md=md[(md.brand!='blank')]

        # print ("\n1-brand is not blank: " +str(md.shape)+'\n')

        ############################―――单子大小与产品单价高低分组―――――――――――

        md['list_price_in_usd'][md.list_price_in_usd>10000000.0]=md['list_price_in_usd']/100
        md['list_price_in_usd'][md.list_price_in_usd==999999.0]=md['tmc_in_usd']*3.3

        md['item_total_list_price']=md['list_price_in_usd']*md['quotation_quantity']
        md['item_total_tmc']=md['tmc_in_usd']*md['quotation_quantity']

        a=md.groupby(['bid_reference_number','create_date']).agg({'item_total_list_price':['sum','count'],'item_total_tmc':['sum']})
        a.columns=['quotation_list_price','quotation_item_num','quotation_tmc']
        md=pd.merge(md,a.reset_index(),on=['bid_reference_number','create_date'],how='left')

        md=md[md.quotation_list_price>=(md['quotation_list_price'].min())]
        md['item_rate']=md['item_total_list_price']/md['quotation_list_price']

        bigger=30000  ##根据业务习惯，看区分大单/小单的quotation_list_price是多少
        md['quotation_big']=(md['quotation_list_price']>=bigger)*1  ##
        md['quotation_big'][md.quotation_list_price>=150000]=2  ##关注15万元以上的quotation

        bigger=1000  ##根据业务习惯，看区分单价高低的产品的价格标准是多少，standard_price_in_usd是多少
        md['product_big']=(md['list_price_in_usd']>=bigger)*1  ##
        md['product_big'][md.list_price_in_usd>=10000]=2

        # print ("\n2-quotation_total_price is not null: " +str(md.shape)+"\n")


        ############################―――――――――――――――

        country=pd.get_dummies(md['country'],prefix='country')
        industry=pd.get_dummies(md['end_customer_nielson_id'],prefix='industry')
        region=pd.get_dummies(md['sub_geo'],prefix='region')
        brand=pd.get_dummies(md['brand'],prefix='brand')
        #prod2=pd.get_dummies(md['prod_brand'],prefix='prod2')
        channel=pd.get_dummies(md['distribution_channel_name'],prefix='channel')
        fulfillment=pd.get_dummies(md['fulfillment_method'],prefix='fulfillment')
        quotation_big=pd.get_dummies(md['quotation_big'],prefix='quotation_big')
        product_big=pd.get_dummies(md['product_big'],prefix='product_big')
        new1=pd.get_dummies(md['deal_new'],prefix='new1')  ##是否新客户
        new2=pd.get_dummies(md['quotation_new'],prefix='new2')
        try:
          md['dol_rate']=1-md['bid_price_in_usd']/(md['list_price_in_usd'])  ##最终客户关注的折扣力度
          md=md[(md.list_price_in_usd>0)]
        except:
          pass

        md['tmc_gap']=1-md['tmc_in_usd']/(md['list_price_in_usd'])  ##产品的利润空间
        md['tmc_gap'][md.list_price_in_usd==0]=np.nan
        md['tmc_gap'][(md.list_price_in_usd==0)&(md.tmc_in_usd==0)]=1

        md['list_price_tmc']=md['list_price_in_usd']/md['tmc_in_usd']
        md['list_price_tmc'][md.tmc_in_usd==0]=np.nan

        md['quotation_tmc_gap']=1-md['quotation_tmc']/(md['quotation_list_price'])  ##产品的利润空间
        md['quotation_tmc_gap'][md.quotation_list_price==0]=np.nan
        md['quotation_tmc_gap'][(md.quotation_list_price==0)&(md.quotation_tmc==0)]=1

        data=pd.concat([quotation_big,product_big,country,industry,brand,channel,md],axis=1)

        alldata=data
        z=alldata[(alldata.list_price_in_usd>0)&(alldata.tmc_in_usd>0)]

        try:
          z=z[(z.bid_status=='Accepted')]##|(z.create_date>sdate)]
          z=z[((z.dol_rate>=0)&(z.margin>=-0.3)&(z.margin<=1.0))]##|(z.create_date>sdate)]
        except:
          pass
        # print ("\n3-bid_status='accepted' & train dol>=0 & -0.3=<margin<=1: " +str(z.shape[0])+"\n")

        return z


    def fun_histop(self, md, sdate):
        dur=0.1

        bins=np.array(np.arange(-0.4, 1.1, dur))
        md['level']=pd.cut(md['dol'],bins)

        u=[['country','end_customer_nielson_id','brand'],['country','brand'],['country']]

        temp=md[['country','end_customer_nielson_id','account_name','brand']].drop_duplicates()
        temp['rank']=1
        # print (temp.shape[0]*5)
        m0=temp
        for i in range(5):
          m1=m0
          m1['rank']=i+1
          temp=temp.append(m1)
        temp=temp[['country','end_customer_nielson_id','account_name','brand','rank']].drop_duplicates()

        for i in range(len(u)):
          x=u[i]

          b00=md[md.create_date<sdate].groupby(x+['level'])['dol'].median()
          b1=md[md.create_date<sdate].groupby(x+['level'])['create_date'].count()
          b2=b1.unstack(len(x))
          b2=b2.div(b2.sum(axis=1),axis=0).stack()
          b=pd.concat([b00,b2],axis=1)
          b.columns=['dol_median','z']

          b10=b1.unstack(len(x)).sum(1).reset_index()
          b10.columns=x+['num']

          b=pd.merge(b.reset_index(),b10,on=x).sort_values(x+['z'],ascending=False)
          b['cum_z']=b.groupby(x)['z'].cumsum()
          b['rank']=b.groupby(x)['z'].rank(ascending=False, method='max')
          b=b[b.num>=200]

          grouped=b.groupby(x)
          top3=grouped.apply(self.get_top,n=5)['dol_median'].reset_index()
          top3['rank']=top3.groupby(x)['dol_median'].rank(ascending=False, method='max')

          m=top3.groupby(x+['rank'])[['dol_median']].sum().unstack(['rank']).fillna(0).stack()
          m.columns=['his_%s' %i]
          m=m.reset_index()

          temp=pd.merge(temp,m,on=x+['rank'],how='left')

        temp['rate']=temp['his_0']
        temp['rate'][-(temp.rate>=-10)]=temp['his_1']
        temp['rate'][-(temp.rate>=-10)]=temp['his_2']
        # print (temp[-(temp.rate>=-10)].shape)

        his_top=temp.groupby(['country','end_customer_nielson_id','account_name','brand','rank'])['rate'].sum().unstack('rank')  #'prod_brand', 'prod_series','prod_family',
        his_top.columns=['his_1','his_2','his_3','his_4','his_5']
        his_top=his_top.reset_index()
        return his_top


    def fun_distribution(self, md, sdate='20180816'):
        u=[['country','end_customer_nielson_id','brand'],['country','brand'],['brand'],['brand','month']]
        i=0

        a=md[(md.create_date<sdate)].groupby(u[i])
        b=a['bid_price_in_usd'].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).reset_index()
        b.columns=u[i]+['a_count','a_avg', 'a_std', 'a_min', 'a_5', 'a_10', 'a_25', 'a_50', 'a_75', 'a_90','a_95', 'a_max']
        x1=b[u[i]+['a_count','a_avg', 'a_std','a_min','a_5','a_10','a_25', 'a_50', 'a_75','a_90','a_95','a_max']][b.a_count>=10].groupby(u[i])['a_avg', 'a_std','a_min','a_5','a_10','a_25', 'a_50', 'a_75','a_90','a_95','a_max'].sum().reset_index()

        a=md[md.create_date<sdate].groupby(u[i])
        b=a['dol_rate'].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).reset_index()
        b.columns=u[i]+['b_count','b_avg', 'b_std', 'b_min', 'b_5', 'b_10', 'b_25', 'b_50', 'b_75', 'b_90','b_95', 'b_max']
        x2=b[u[i]+['b_count','b_avg', 'b_std','b_min', 'b_5','b_10','b_25', 'b_50', 'b_75','b_90','b_95','b_max']][b.b_count>=10].groupby(u[i])['b_avg', 'b_std','b_min', 'b_5','b_10','b_25', 'b_50', 'b_75','b_90','b_95','b_max'].sum().reset_index()

        a=md[md.create_date<sdate].groupby(u[i])
        b=a['margin'].describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]).reset_index()
        b.columns=u[i]+['c_count','c_avg', 'c_std', 'c_min', 'c_5', 'c_10', 'c_25', 'c_50', 'c_75', 'c_90','c_95', 'c_max']
        x3=b[u[i]+['c_avg', 'c_std','c_count','c_min', 'c_5','c_10','c_25', 'c_50', 'c_75','c_90','c_95','c_max']][b.c_count>=10].groupby(u[i])['c_avg', 'c_std','c_min', 'c_5','c_max'].sum().reset_index()
        return x1,x2,x3


    def model_train(self,x_train,y_train,data,num_boost=10,outdir='x'):

        dtrain=xgb.DMatrix(x_train,label=y_train)

        params={'booster':'gbtree',
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'max_depth':4,
            'lambda':10,
            'subsample':0.75,
            'colsample_bytree':0.75,
            'min_child_weight':2,
            'eta': 0.025,
            'seed':2000,
            'nthread':8,
            'silent':1,
            'scale_pos_weight':2}

        watchlist = [(dtrain,'train')]
        bst=xgb.train(params,dtrain,num_boost_round=num_boost,evals=watchlist)
        self.xgbmodel = bst
#         joblib.dump(bst, outdir+'bst_NA.pkl')

        ############# RandomForest Model train ##################

        regr_rf= RandomForestRegressor(n_estimators=50,random_state=2,min_samples_split=20,min_samples_leaf=4,max_features=0.2,n_jobs=4)
        regr_rf.fit(x_train, y_train)
        self.rfmodel = regr_rf
#         xpred2=regr_rf.predict(x_train)
#         joblib.dump(regr_rf, outdir+'rf_NA.pkl')

        importance=regr_rf.feature_importances_
        dd=pd.concat([DataFrame(data.iloc[:,:-3].columns),DataFrame(importance)],axis=1)
        dd.columns=['feature','importance']
        dd=dd.sort_values(['importance'],ascending=False)

        return dd,bst,regr_rf


    def pred_res(self,m):
        m['price_pred']=(1-m['pred_dol'])*m['list_price_in_usd']

        m['gap']=m['list_price_in_usd']*0.05
        m['gap'][m.list_price_in_usd<=1500]=m['list_price_in_usd']*0.075
        m['price_min']=m['price_pred']-m['gap']
        m['price_max']=m['price_pred']+m['gap']
        m['price_max'][m.price_max>m.list_price_in_usd]=m['list_price_in_usd']

        m['price_min'][m.price_pred<50]=m['price_min']-1
        m['price_max'][m.price_pred<50]=m['price_max']+1

        m['list']=m['list_price_in_usd']*m['quotation_quantity']
        m['min']=m['price_min']*m['quotation_quantity']
        m['pred']=m['price_pred']*m['quotation_quantity']
        m['max']=m['price_max']*m['quotation_quantity']

        m=m.drop(['gap'], 1)
        m['create_date']=m['create_date'].map(lambda s: str(s)[:10])
        m['create_date']=pd.to_datetime(m['create_date'])

        m=m.sort_values(['create_date','bid_reference_number','bid_reference_number_item'])
        try:
            m['bid_price_range']=np.nan
            m['bid_price_range'][(m.bid_price_in_usd<=1000)&(m.bid_price_in_usd>0)]='0-1000'
            m['bid_price_range'][(m.bid_price_in_usd<=10000)&(m.bid_price_in_usd>1000)]='1000-10000'
            m['bid_price_range'][(m.bid_price_in_usd>10000)]='10000-more'

            m['abs_dol_gap']=abs(m['dol_rate']-m['pred_dol'])
            m['acc']=1-abs(m.bid_price_in_usd-m.price_pred)/m.bid_price_in_usd

            m['flag']='bid_price_isin'
            m['flag'][(m.bid_price_in_usd<m.price_min)]='pred_higher_price'
            m['flag'][(m.bid_price_in_usd>m.price_max)]='pred_lower_price'

            m['bid']=m['bid_price_in_usd']*m['quotation_quantity']
            #print (m['acc'].mean())
        except:
            pass
        return m
    
    
    def merge_data(self,origin, result):
        # result table to origin table's meta
        domain = origin.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        new_domain = Domain(attributes, classes, meta_attrs+result.domain.variables)
        new_table = Table.from_table(new_domain, origin)
        for attr in result.domain:
            new_table.get_column_view(attr)[0][:] = result.get_column_view(attr)[0][:]
        return new_table


    def run(self):
        md = pd.concat([table2df(self.train_raw), table2df(self.test_raw)])
        
        # get label & sentence columns name
        if self.train_raw.domain.class_var.name != self.test_raw.domain.class_var.name:
            raise Error('Label for training data and test data must be the same')
            
        self.label_name = self.train_raw.domain.class_var.name
        
        if self.label_name == 'dol':
            self.label_name = 'dol_rate'

        mdcols=['account_name', 'bid_price_in_usd',
               'bid_reference_number', 'bid_reference_number_item', 'bid_status',
               'brand', 'country', 'create_date','distribution_channel_name',
               'dol', 'end_customer_nielson_id', 'fulfillment_method', 
               'list_price_in_usd', 'margin', 'net_revenue_in_usd',
               'opportunity', 'quotation_quantity',
               'sub_geo','tmc_in_usd'] 
        self.features = mdcols
        
        if not all([True if col in md.columns else False for col in self.features]):
            raise Error('Missing required features.')
            
            
        selfdir='./'                   ##本地存放文件夹位置
        dataname='data_samples'        ##数据csv名称
        sdate=md['create_date'].sort_values().iloc[round(len(md)*0.8)]         ##划分区隔train与test数据的日期
        num_boost=50                    ##xgb参数
        
        datadir=selfdir
        funcdir=selfdir
        outdir=selfdir

        md=md.dropna(how='all',axis='columns')

        # data pre-process
        md=self.pre_process(md, sdate)

        his_top=self.fun_histop(md, sdate)
        md=pd.merge(his_top,md,on=['country','end_customer_nielson_id','account_name','brand'],how='right')

        x1,x2,x3=self.fun_distribution(md,sdate)
        u=[['country','end_customer_nielson_id','brand'],['country','brand'],['brand'],['brand','month']]
        md=pd.merge(x1,md,on=u[0])
        md=pd.merge(x2,md,on=u[0])
        md=pd.merge(x3,md,on=u[0])
        md2=md[(md.tmc_in_usd>0)&(md.quotation_quantity>0)&((md.his_5>-1))]
        # print ("\ndata prepare finished: " +str(md2.shape)+"\n")

        #############  train & test data #################

        col=DataFrame(list(md2.columns),columns=['x'])
        col['m']=col['x'].map(lambda s: s.split('_')[0])
        col=list(col['x'][(col.m.isin(['a','b','c','brand','channel','country','his','industry']))&(- col.x.isin(['brand','channel','country','industry']))].unique())
        data=md2[col+['quotation_big_0','quotation_big_1', 'quotation_big_2', 'product_big_0', 'product_big_1',
               'product_big_2','quotation_new', 'deal_num', 'deal_new', 'quotation_list_price',
               'quotation_tmc', 'quotation_quantity', 'item_rate', 'tmc_gap',
               'item_total_list_price', 'item_total_tmc', 'list_price_tmc',
               'quotation_tmc_gap', 'list_price_in_usd', 'tmc_in_usd','create_date','bid_reference_number']]
        
        label_cols = ['dol_rate','create_date','bid_reference_number','bid_reference_number_item', 'quotation_quantity','list_price_in_usd','tmc_in_usd','net_revenue_in_usd','dol','bid_price_in_usd','account_name','approving_status','approval_result', 'fulfillment_method','sub_geo','country','brand','distribution_channel_name','end_customer_nielson_id']
        
        if self.label_name in label_cols:
            label=md2[label_cols]
        else:
            label=md2[[self.label_name] + [label_cols]]

        try:
            md[self.label_name] = md[self.label_name].astype(float)
        except Exception as e:
            print(e)
            raise Error('Missing Label column or Label column has incorrect data type')
            
        x_train=data[data.create_date<sdate].iloc[:,:-2]
        self.train_col=x_train.columns
        x_train=x_train.values
        # print ("\nModel selected %s feature:\n" %(x_train.shape[1]))
        y_train=label[label.create_date<sdate].loc[:,self.label_name].values
        x_test=data[(data.create_date>=sdate)].iloc[:,:-2].values
        y_test=label[(label.create_date>=sdate)].loc[:,self.label_name].values
        
        print(len(x_test))
        
        var_lst = [ContinuousVariable(x) for x in self.train_col]
        self.train_data = Table.from_numpy(Domain(var_lst), x_train)
        self.test_data = Table.from_numpy(Domain(var_lst), x_test)
        # print ("train data shape: " +str(x_train.shape))
        # print ("test data shape: " +str(x_test.shape)+"\n")
        
        ############# Model train #################
        t1=time.time()

        featrue_importance,bst_model,regr_rf_model=self.model_train(x_train,y_train,data,num_boost,outdir)

        # featrue_importance.to_csv(outdir+'feature_importance.csv',index=False)
        # print ("Top 10 feature:")
        # print (featrue_importance.head(10).reset_index(drop=True))
        #
        # print ("\nModel training time:  %.2fs" %(time.time()-t1))

        ############# Model Test result  #################
        ypred1=self.xgbmodel.predict(xgb.DMatrix(x_test))
        ypred2=self.rfmodel.predict(x_test)
        y_pred=ypred1*0.5+ypred2*0.5

        y=label[label.create_date>=sdate].iloc[:,:].values
        m=pd.concat([DataFrame(y),DataFrame(y_pred)],axis=1)
        m.columns=list(label.columns)+['pred_dol']

        res=self.pred_res(m)
        res['acc']=res['acc'].astype(float)
        # print ('Train END\n')
        d_item=(res['acc']*res['bid']/(res.bid.sum())).sum()
        print('Test item weighted accuracy: %.2f%%'  %(d_item*100))

        quot_res=res.groupby(['bid_reference_number'])['bid','pred','net_revenue_in_usd'].sum()
        quot_res['acc']=1-abs(quot_res.bid-quot_res.pred)/quot_res.bid
        d_quote=(quot_res['acc']*quot_res['bid']/(quot_res.bid.sum())).sum()
        print('Test quotation weighted accuracy: %.2f%%'  %(d_quote*100))

        metas = [ContinuousVariable('predict')]
        
        cols = [np.array(y_pred).reshape(-1,1)]
        res = Table.from_numpy(Domain(metas), y_pred.reshape(-1,1))
        final_result = self.merge_data(self.test_data, res)

        self.send("News", final_result)
        self.send("Metric Score", None)
        self.send("Metas", metas)
        self.send("Columns", cols)
        
        results = Results(self.test_data, store_data=True)
        results.folds = None
        results.row_indices = np.arange(len(y_test))
        results.actual = y_test
        results.predicted = y_pred
        results.probabilities = y_pred
        self.send("Predictions", y_pred)
        self.send("Evaluation Results", results)
        self.send("Metric", MetricType.ROOT_MEAN_SQUARED_ERROR)
                                 
        metric_frame = MetricFrame([[d_item], [d_quote]], index=["Item level", "Quote level"],
                           columns=[MetricType.ROOT_MEAN_SQUARED_ERROR.name])
        self.send("Metric Score", metric_frame)

    def rerun(self):
        self.test_data = self.test_raw
        md = table2df(self.test_raw)

        try:
            assert(all([True if col in md.columns else False for col in self.features]))
        except Exception as e:
            missing_cols = [col for col in self.features if col not in md.columns]
            raise Error('Missing required features: {}'.format(missing_cols))
        
        sdate=md['create_date'].max()         ##划分区隔train与test数据的日期
        md=md.dropna(how='all',axis='columns')

        # data pre-process
        md=self.pre_process(md,sdate)

        his_top=self.fun_histop(md,sdate)
        md=pd.merge(his_top,md,on=['country','end_customer_nielson_id','account_name','brand'],how='right')

        x1,x2,x3=self.fun_distribution(md,sdate)
        u=[['country','end_customer_nielson_id','brand'],['country','brand'],['brand'],['brand','month']]
        md=pd.merge(x1,md,on=u[0])
        md=pd.merge(x2,md,on=u[0])
        md=pd.merge(x3,md,on=u[0])
        md2=md[(md.tmc_in_usd>0)&(md.quotation_quantity>0)&((md.his_5>-1))]
        # print ("\ndata prepare finished: " +str(md2.shape)+"\n")

        #############  train & test data #################

        col=DataFrame(list(md2.columns),columns=['x'])
        col['m']=col['x'].map(lambda s: s.split('_')[0])
        col=list(col['x'][(col.m.isin(['a','b','c','brand','channel','country','his','industry']))&(- col.x.isin(['brand','channel','country','industry']))].unique())
        data=md2[col+['quotation_big_0','quotation_big_1', 'quotation_big_2', 'product_big_0', 'product_big_1',
               'product_big_2','quotation_new', 'deal_num', 'deal_new', 'quotation_list_price',
               'quotation_tmc', 'quotation_quantity', 'item_rate', 'tmc_gap',
               'item_total_list_price', 'item_total_tmc', 'list_price_tmc',
               'quotation_tmc_gap', 'list_price_in_usd', 'tmc_in_usd','create_date','bid_reference_number']]
      
        x_test = data[(data.create_date>=sdate)].iloc[:,:-2].values
        
        var_lst = [ContinuousVariable(x) for x in self.train_col]
        self.test_data = Table.from_numpy(Domain(var_lst), x_test)
        
        ypred1=self.xgbmodel.predict(xgb.DMatrix(x_test))
        
        ypred2=self.rfmodel.predict(x_test)
        y_pred=ypred1*0.5+ypred2*0.5
        
        metas = [ContinuousVariable('predict')]
        
        cols = [np.array(y_pred).reshape(-1,1)]
        res = Table.from_numpy(Domain(metas), y_pred.reshape(-1,1))
        final_result = self.merge_data(self.test_data, res)

        self.send("News", final_result)
        self.send("Metric Score", None)
        self.send("Metas", metas)
        self.send("Columns", cols)
        
