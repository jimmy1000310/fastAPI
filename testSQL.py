import psycopg2
import numpy as np

conn = psycopg2.connect(database = 'postgres' ,user = 'postgres',password='123456',port='5432')

curs=conn.cursor()

# sql = 'select * from weather'   #select(讀取) *(所有) from(在) weather(檔案名稱)
# curs.execute(sql)               #執行SQL指令  
# curs.execute()
# data=curs.fetchall()            #將資料取出丟入data
# print(data)
# curs.close()
# conn.close()

# #展示現有資料庫
# sql="SELECT tablename FROM PG_TABLES WHERE SCHEMANAME = 'public';"
# curs.execute(sql)
# print("現有結果{}".format( curs.fetchall()))
# #創建Person 資料庫
# sql="""CREATE TABLE "Person"("ID" SERIAL PRIMARY KEY NOT NULL, "Name" TEXT, "Age" INT)"""
# curs.execute(sql)
# #展示創建後之資料庫
# sql="SELECT tablename FROM PG_TABLES WHERE SCHEMANAME = 'public';"
# curs.execute(sql)
# print("創建後{}".format( curs.fetchall()))
# #刪除新創之資料庫
# sql="""DROP TABLE "Person";"""
# curs.execute(sql)
# #展示刪除後資料庫
# sql="SELECT tablename FROM PG_TABLES WHERE SCHEMANAME = 'public';"
# curs.execute(sql)
# print("創建後{}".format( curs.fetchall()))


# #刪除新創之資料庫
# sql="""DROP TABLE "iris";"""
# curs.execute(sql)
#展示現有資料庫
sql="SELECT tablename FROM PG_TABLES WHERE SCHEMANAME = 'public';"
curs.execute(sql)
print("現有結果{}".format( curs.fetchall()))
#建立花資料庫
sql="""CREATE TABLE "iris" ("Speal Leng" real,"Speal Width" real,"Petel Leng" real,"Petal Width" real,"Species" varchar(80));"""   
curs.execute(sql)
#展示現有資料庫
sql="SELECT tablename FROM PG_TABLES WHERE SCHEMANAME = 'public';"
curs.execute(sql)
print("現有結果{}".format( curs.fetchall()))
# 將txt檔案轉成資料庫
sql="""COPY iris FROM 'D:\python/fastAPI/iris.txt' (DELIMITER(','));"""
curs.execute(sql)
# sql="""INSERT INTO iris VALUES (5.1,3.5,1.4,0.2,'Iris-setosa');"""
# curs.execute(sql)
# 提取出資料庫資料
sql = 'select * from iris;'
curs.execute(sql) 
data=curs.fetchall() 
print(type(data[0]))
a=np.array(data)
print(a[0][0])
curs.close()
conn.close()

