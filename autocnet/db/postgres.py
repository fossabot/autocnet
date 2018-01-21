from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer,Float
from geoalchemy2 import Geometry

Base = declarative_base()

attr_dict = {'__tablename__':None,
             '__table_args__': {'useexisting':True},    
             'id':Column(Integer, primary_key=True, autoincrement=True)
             'name':Column(String),
             'path':Column(String),
             'footprint':Column(Geometry('POLYONG')),
             'keypoint_path':Column(String),
             'nkeypoints':Column(Integer),
             'kp_min_x':Column(Float),
             'kp_max_x':Column(Float),
             'kp_min_y':Column(Float),
             'kp_max_y':Column(Float)}

def create_table_cls(name, clsname):
    attrs = attr_dict
    attrs['__tablename__'] = name
    return type(clsname, (Base,), attrs)
