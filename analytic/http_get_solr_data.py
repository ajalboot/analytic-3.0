# from urllib.request import urlopen
# import json
#
# # server_name = 'http://206.167.183.217:8983/solr/'
# server_name = 'http://129.173.67.59:8983/solr/'
#
# def get_tids_in_database(database):
#     url = server_name+str(database)+"/select?fl=tid&q=tid:*&rows=1000000&wt=json"
#     # print url
#     get_data = urlopen(url).read()
#
#     point_line_object = json.loads(get_data)['response']['docs']
#     tids_array = []
#     for elem in point_line_object:
#         tids_array.append(elem['tid'])
#     # print sorted(tids_array)
#     return sorted(tids_array)
#
#
# def create_geometry(array_of_values):
#     bidim_array = []
#     index = 0
#     max_index = len(array_of_values)
#     # print max_index
#     while index < max_index:
#         a = [array_of_values[index], array_of_values[index+1]]
#         bidim_array.append(a)
#         index += 2
#
#     return bidim_array
#
#
# def get_geojson_layer(database, tid):
#
#     geojson = {}
#     geojson['type'] = 'Feature'
#     geojson['tid'] = tid
#     geojson['properties'] = {}
#     geojson['geometry'] = {}
#     geojson['geometry']['type'] = 'LineString'
#     geojson['points'] = {}
#
#     url = server_name+str(database)+"/select?q=tid:"+str(tid)+"&rows=1&wt=json"
#     # print url
#     get_data = urlopen(url).read()
#
#     point_line_object = json.loads(get_data)['response']['docs'][0]
#     geojson['geometry']['coordinates'] = create_geometry(point_line_object['geometry'])
#     # print point_line_object
#
#     properties_keys = point_line_object.keys()
#     # print properties_keys
#
#     tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]
#     pf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('pf_')]
#
#     for tf in tf_fields:
#         geojson['properties'][tf] = point_line_object['tf_'+str(tf)]
#
#     for pf in pf_fields:
#         geojson['points'][pf] = point_line_object['pf_'+str(pf)]
#
#     # print geojson
#     return geojson
#
#
# def get_all_trajectory_features(database):
#     map_tid_feature = {}
#     get_data = urlopen(
#         server_name + database + \
#         "/select?&q=tid:*&rows=1000000&wt=json").read()
#     all_data_lines = json.loads(get_data)['response']['docs']
#     # get keys
#     properties_keys = all_data_lines[0].keys()
#     tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]
#
#     for line in all_data_lines:
#         # print line
#         # data = {}
#         # data[line['tid']] = line['tid']
#
#         fields = {}
#         for tf in tf_fields:
#             fields[tf] = line['tf_'+str(tf)]
#
#         # data[line['tid']] = fields
#         # map_tid_feature.append(data)
#         map_tid_feature[line['tid']] = fields
#
#     # print map_tid_feature
#     return  map_tid_feature
#
#
# def get_trajectories_geojson(database, tids):
#     out = ''
#     for tid in tids:
#         out += json.dumps(get_geojson_layer(database, tid)) + '\n'
#     # print out
#     return out
#
# # get_tids_in_database('fishingvessels')
# # get_geojson_layer('fishingvessels', 2)
# # get_all_trajectory_features('fishingvessels')
from urllib.request import urlopen
import json
import requests


# def get_tids_in_database(database):
#     print("bbbbb")
#     url = "http://localhost:8080/solr/" + str(database) + "/select?fl=tid&q=tid:*&rows=1000000&wt=json"
#     try:
#         get_data = urlopen(url).read()
#         print("AAAAA")
#         point_line_object = json.loads(get_data)['response']['docs']
#         tids_array = []
#         for elem in point_line_object:
#             tids_array.append(elem['tid'])
#         return sorted(tids_array)
#     except Exception as e:
#         print("An error occurred:", e)
#         return []

def get_tids_in_database(database):
    try:
        with open(database, 'r') as file:
            data = json.load(file)

            tids_array = []
            for elem in data:
                tids_array.append(elem['line']['tid'])
            return sorted(tids_array)
    except Exception as e:
        print("An error occurred:", e)
        return []

def create_geometry(array_of_values):
    bidim_array = []
    index = 0
    max_index = len(array_of_values)
    # print max_index
    while index+2 < max_index:
        a = [array_of_values[index], array_of_values[index+1]]
        bidim_array.append(a)
        index += 2

    return bidim_array


def get_geojson_layer(database, tid_to_find):
    # database = "./data/animals.json"
    geojson = {}
    geojson['type'] = 'Feature'
    geojson['tid'] = tid_to_find
    geojson['properties'] = {}
    geojson['geometry'] = {}
    geojson['geometry']['type'] = 'LineString'
    geojson['points'] = {}

    # Replace the Solr URL with the URL of the desired data source
    # url = "http://localhost:8080/{}/{}".format(database, tid)

    try:
        # Load the JSON data
        with open(database, 'r') as file:
            data = json.load(file)

        for tid, item in enumerate(data):
            if item['line']['tid'] == tid_to_find:
                # print('Found at index:', tid)
                break

        # print(data[tid]['line']['geometry']['coordinates'])
        # Filter lines and points based on 'tid' values
        point_line_object = data[tid]['line']
        points_object = data[tid]['points']
        # geojson['geometry']['coordinates'] = create_geometry(point_line_object['geometry']['coordinates'])
        geojson['geometry']['coordinates'] = point_line_object['geometry']['coordinates']

        properties_keys = point_line_object['properties']

        # tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]
        # pf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('pf_')]

        for tf in properties_keys:
            geojson['properties'][tf] = point_line_object['properties'][str(tf)]

        for pf in range(len(points_object)):
            geojson['points'][pf] = points_object[pf]

        # print("-----------------------------------------------------------------------------------------------------------")
        # print(type(geojson))
        # print(geojson)
        return geojson

    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occurred during the request
        print(f"An error occurred while fetching data: {e}")
        return None


# def get_geojson_layer(database, tid):
#
#     geojson = {}
#     geojson['type'] = 'Feature'
#     geojson['tid'] = tid
#     geojson['properties'] = {}
#     geojson['geometry'] = {}
#     geojson['geometry']['type'] = 'LineString'
#     geojson['points'] = {}
#
#     url = "http://localhost:8080/solr/"+str(database)+"/select?q=tid:"+str(tid)+"&rows=1&wt=json"
#     # print url
#     get_data = urlopen(url).read()
#
#     point_line_object = json.loads(get_data)['response']['docs'][0]
#     geojson['geometry']['coordinates'] = create_geometry(point_line_object['geometry'])
#     # print point_line_object
#
#     properties_keys = point_line_object.keys()
#     # print properties_keys
#
#     tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]
#     pf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('pf_')]
#
#     for tf in tf_fields:
#         geojson['properties'][tf] = point_line_object['tf_'+str(tf)]
#
#     for pf in pf_fields:
#         geojson['points'][pf] = point_line_object['pf_'+str(pf)]
#
#     # print geojson
#     return geojson


def get_all_trajectory_features(database):
    map_tid_feature = {}
    # database = "./data/animals.json"

    try:
        with open(database, 'r') as file:
            all_data_lines = json.load(file)

        point_line_object = all_data_lines[0]['line']
        tf_fields = point_line_object['properties']


        # properties_keys = all_data_lines[0].keys()
        # tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]

        for line in all_data_lines:
            fields = {}
            for tf in tf_fields:
                # print(line['line']['properties'])
                # print(line[str(tf)])
                fields[tf] = line['line']['properties']

            # print(line['line']['tid'])
            map_tid_feature[line['line']['tid']] = fields

        return map_tid_feature

    except Exception as e:
        print("An error occurred:", e)
        return {}


def get_trajectories_geojson(database, tids):
    out = ''
    for tid in tids:
        x = json.dumps(get_geojson_layer(database, tid), default=float)
        out += x + '\n'
    # print("-------------------------------")
    # print(out)
    return out

# get_tids_in_database('fishingvessels')
# get_geojson_layer('fishingvessels', 2)
# get_all_trajectory_features('fishingvessels')