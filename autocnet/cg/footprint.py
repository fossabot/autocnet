import pyproj
import ogr

def generate_latlon_footprint(camera, nnodes=5, semi_major=3396190, semi_minor=3376200):
    ecef = pyproj.Proj(proj='geocent', a=semi_major, b=semi_minor)
    lla = pyproj.Proj(proj='latlon', a=semi_major, b=semi_minor)

    isize = camera.imagesize[::-1]
    x = np.linspace(0,isize[1], 10)
    y = np.linspace(0,isize[0], 10)
    boundary = [(i,0.) for i in x] + [(isize[1], i) for i in y[1:]] +\
               [(i, isize[0]) for i in x[::-1][1:]] + [(0.,i) for i in y[::-1][1:]]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for i in boundary:
        gnd = camera.imageToGround(*i, 0)
        lons, lats, alts = pyproj.transform(ecef, lla, gnd[0], gnd[1], gnd[2])
        ring.AddPoint(lons, lats)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

def generate_bodyfixed_footprint(camera, nnodes=5):
    isize = camera.imagesize[::-1]
    x = np.linspace(0,isize[1], 10)
    y = np.linspace(0,isize[0], 10)
    boundary = [(i,0.) for i in x] + [(isize[1], i) for i in y[1:]] +\
               [(i, isize[0]) for i in x[::-1][1:]] + [(0.,i) for i in y[::-1][1:]]
    ring = ogr.Geometry(ogr.wkbLinearRing)

    for i in boundary:
        gnd = camera.imageToGround(*i, 0)
        ring.AddPoint(gnd[0], gnd[1], gnd[2])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly
