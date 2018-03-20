# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import sh
#import MySQLdb
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
  
  #output the info messages
  logging.getLogger().setLevel(logging.INFO)	
 
  #copying the file from S3 to S3
  try:
    os.system("aws s3 cp {s3_bucket_input} {s3_bucket_output}".format(s3_bucket_input=args.s3_bucket_input, s3_bucket_output=args.s3_bucket_output))
    os.system("aws s3 ls s3://tf-bucket-dev/OUTPUT/")
    logging.info("File copied to S3 location: {s3_bucket_output}".format(s3_bucket_output=args.s3_bucket_output))
  except:
    logging.error("File is not copied to S3")

    
  #saving output to the postgres db
  try:                 
    engine = create_engine("postgres://{user}:{passwd}@{host}/{db}".format(host=args.db_host,    # your host, usually localhost
                                                                                   user=args.db_user,         # your username
                                                                                   passwd=args.db_pass,  # your password
                                                                                   db=args.db_name))       # name of the data base)

    df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('BCDE'))
    df.to_sql(con=engine, name='table_name_postgresql', if_exists='replace')
    logging.info("Results saved to the DB")

  except:
    logging.error("Cannot save to DB")
    
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   