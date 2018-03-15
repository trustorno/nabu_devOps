# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import sh
import MySQLdb
from subprocess import call
from pandas.io import sql
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--s3_bucket_input', default = '123')
  parser.add_argument('-o', '--s3_bucket_output', default = '123')
  parser.add_argument('-d', '--db_host', default = '123')
  parser.add_argument('-u', '--db_user', default = '')
  parser.add_argument('-p', '--db_pass', default = '')
  parser.add_argument('-n', '--db_name', default = '')
  return parser.parse_args()	

if __name__ == "__main__":	
  args = parse_args()
	
 
  #copying the file from S3 to S3
  #try:
  #s3 = sh.bash.bake("aws s3")
  #s3.cp(args.s3_bucket_input, args.s3_bucket_output)
  #os.system("aws s3 cp {s3_bucket_input} {s3_bucket_output}".format(s3_bucket_input=args.s3_bucket_input, s3_bucket_output=args.s3_bucket_output))
  #os.system("aws s3 ls s3://tf-bucket-dev/OUTPUT/")
  #logging.warning("File copied to S3 location: {s3_bucket_output}".format(s3_bucket_output=args.s3_bucket_output))
  #except:
  #  logging.error("File is not copied to S3")
    
    
  #saving output to the mysql db
  #try:
  db = MySQLdb.connect(host=args.db_host,    # your host, usually localhost
                       user=args.db_user,         # your username
                       passwd=args.db_pass,  # your password
                       db=args.db_name)        # name of the data base
                       
  engine = create_engine("mysql+mysqldb://{user}:{passwd}@{host}/{db}".format(host=args.db_host,    # your host, usually localhost
                                                                                   user=args.db_user,         # your username
                                                                                   passwd=args.db_pass,  # your password
                                                                                   db=args.db_name))       # name of the data base)

    
  # you must create a Cursor object. It will let
  #  you execute all the queries you need
  
  df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
  df.to_sql(con=engine, name='table_name_for_df', if_exists='replace')
  db.commit()
  
  
  print "test"
  
  # Use all the SQL you like
  cur = db.cursor()
  cur.execute("show tables")
  
  # print all the first cell of all the rows
  for row in cur.fetchall():
      logging.warning(row[0])
  
  db.close()  
  logging.warning("Data saved to DB")
  #except:
  #  logging.error("Cannot save to DB")
    
   