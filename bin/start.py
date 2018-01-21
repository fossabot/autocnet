import autocnet
import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from autocnet.db.postgres import create_table_cls

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--database', help='The database url to connect to.')
    parser.add_argument('-d', '--name', help='The name of the database')
    parser.add_argument('table', help='The database table to connect to.')
    return vars(parser.parse_args())

def setup_db(database_url, datanase_name):
    engine = create_engine(database_url + '/' + database_name)
    session = Session(bind=engine)

    return session



if __name__ == '__main__':
    args = parse_args()
    database_url = args.pop('database')
    datbase_name = args.pop('name')
    table = args.pop('table')

    keypoint_table = create_table_cls('keypoints', 'keypoint_table')
    
    session = setup_db(database_url, database_name)
