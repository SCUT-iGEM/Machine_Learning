import pandas as pd
import numpy as np
import os

def simple_data():
    f_path = "I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM!\不过过是重头再来\数据数据数据数据\GSE104878_pTpA_random_design_tiling_etc_YPD_expression.txt"

    df_raw = pd.read_csv(f_path,sep='\t',low_memory=False,encoding='utf-8')
    df_raw.columns = ['sequence','value']
    df_raw.dropna(subset=['value'],inplace=True)


    # 查看最大最小值
    df_raw['value']=df_raw['value'].map(float)
    max_value = np.max(df_raw['value'].tolist())
    min_value = np.min(df_raw['value'].tolist())
    print('max',max_value,'min',min_value)
    print(df_raw.shape)

    '''
    df1 = df_raw[df_raw['value']>=13]
    print('14~16',df1.shape)
    df1.to_csv()
    df2 = df_raw[df_raw['value']>=10.5]
    df2 = df2[df2['value']<13]
    print('10~13',df2.shape)
    
    
    df3 = df_raw[df_raw['value']>=7]
    df3 = df3[df3['value']<10.5]
    print('7~10',df3.shape)
    
    df4 = df_raw[df_raw['value']>=0]
    df4 = df4[df4['value']<7]
    print("0~7",df4.shape)
    '''

    df1 = df_raw[df_raw['value']>=12]
    print('14~16',df1.shape)
    df1.to_csv()
    df2 = df_raw[df_raw['value']>=9]
    df2 = df2[df2['value']<12]
    print('10~13',df2.shape)


    df3 = df_raw[df_raw['value']>=0]
    df3 = df3[df3['value']<9]
    print('7~10',df3.shape)


    out_path = r'I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM\!\不过过是重头再来\数据数据数据数据\large_txt_data'
    def write_txt(df,group_name):
        with open(os.path.join(out_path,'{}.txt'.format(group_name)),'a+',newline="")as f:
            for line in df.values:
                f.write(line[0]+'\n')

    write_txt(df1,'high')
    write_txt(df2,'mid')
    write_txt(df3,'low')
    # write_txt(df4,'extreme low')

def so_large_data():
    f_path = r'I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM!\不过过是重头再来\数据数据数据数据\GSE104878_20170811_average_promoter_ELs_per_seq_OLS_Glu_goodCores_ALL.txt'

    df = pd.read_csv(f_path, sep='\t', low_memory=False, encoding='utf-8',chunksize=300000)
    size1,size2,size3,size4,size5,size6,size7 = 0,0,0,0,0,0,0
    for df_raw in df:
        df_raw.columns = ['whatever','sequence', 'value','what','ever']
        df_raw = df_raw[['sequence','value']]
        df_raw.dropna(subset=['value'], inplace=True)

        # 查看最大最小值
        df_raw['value'] = df_raw['value'].map(float)
        max_value = np.max(df_raw['value'].tolist())
        min_value = np.min(df_raw['value'].tolist())
        print('max', max_value, 'min', min_value)
        print(df_raw.shape)

        '''
        df1 = df_raw[df_raw['value']>=13]
        print('14~16',df1.shape)
        df1.to_csv()
        df2 = df_raw[df_raw['value']>=10.5]
        df2 = df2[df2['value']<13]
        print('10~13',df2.shape)
        df3 = df_raw[df_raw['value']>=7]
        df3 = df3[df3['value']<10.5]
        print('7~10',df3.shape)
        
        df4 = df_raw[df_raw['value']>=0]
        df4 = df4[df4['value']<7]
        print("0~7",df4.shape)
        
        
        df1 = df_raw[df_raw['value'] >= 4]
        size1+=df1.shape[0]
        df1.to_csv()
        df2 = df_raw[df_raw['value'] >= 1.5]
        df2 = df2[df2['value'] < 4]
        size2+=df2.shape[0]

        df3 = df_raw[df_raw['value'] >= 0]
        df3 = df3[df3['value'] < 1.5]
        size3+=df3.shape[0]
        print(size1,size2,size3)
        
        '''
        df1 = df_raw[df_raw['value'] >= 5]
        size1+=df1.shape[0]
        df1.to_csv()

        df2 = df_raw[df_raw['value'] >= 4]
        df2 = df2[df2['value'] < 5]
        size2+=df2.shape[0]

        df3 = df_raw[df_raw['value'] >= 3.5]
        df3 = df3[df3['value'] < 4]
        size3+=df3.shape[0]

        df4 = df_raw[df_raw['value'] >= 3]
        df4 = df4[df4['value'] < 3.5]
        size4+=df4.shape[0]

        df5 = df_raw[df_raw['value'] >= 2.5]
        df5 = df5[df5['value'] < 3]
        size5 += df5.shape[0]

        df6 = df_raw[df_raw['value'] >= 1]
        df6 = df6[df6['value'] < 2]
        size6+=df6.shape[0]

        df7 = df_raw[df_raw['value'] >= 0]
        df7 = df7[df7['value'] < 1]
        size7+=df7.shape[0]
        print(size1,size2,size3,size4,size5,size6,size7)

        # out_path = r'I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM\!\不过过是重头再来\数据数据数据数据\large_txt_data'
        out_path = r'I:\坠机堡垒装不下\E盘Work\SCUT\BioS\iGEM\!\不过过是重头再来\数据数据数据数据\split_txt_data'

        def write_txt(df, group_name,out_path):
            with open(os.path.join(out_path, '{}.txt'.format(group_name)), 'a+', newline="")as f:
                for line in df.values:
                    f.write(line[0] + '\n')

    #     write_txt(df1, 'very_high',out_path)
    #     write_txt(df2, 'high',out_path)
    #     write_txt(df3, 'rare_high',out_path)
    #     write_txt(df4, 'mid',out_path)
    #     write_txt(df5, 'rare_low',out_path)
    #     write_txt(df6, 'low',out_path)
    #     write_txt(df7, 'very_low',out_path)
    #     # write_txt(df4,'extreme low')
    # total_size = size1+size2+size3
    # print('total_size',total_size)
so_large_data()