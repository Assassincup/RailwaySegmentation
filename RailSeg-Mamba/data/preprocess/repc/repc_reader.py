import struct
from typing import List

# repc文件文件头结构



class LASheader:
    file_signature = ''  # 文件签名4
    file_id = 0  # 文件编号
    major = 0  # 文件主版本号
    minor = 0  # 文件副版本号
    hardware_system = ''  # 采集硬件系统名称、版本
    software_system = ''  # 采集软件系统名称、版本
    railway_name = ''  # 线路名
    railway_num = ''  # 线路编号108
    railway_direction = ''  # 行别
    mile_direction = ''  # 增、减里程
    start_mileage = 0  # 开始里程
    train_no = ''  # 检测车号
    position = ''  # 位端
    time = 0  # 开始时间（1970年1月1日0点0分距开始时间的毫秒数）
    header_size = 0  # 文件头长度
    point_data_size = ''  # 单个点数据字节数
    point_number = 0  # 点总数量46
    x_scale_factor = 0  # X尺度因子
    y_scale_factor = 0  # Y尺度因子
    z_scale_factor = 0  # Z尺度因子
    x_offset = 0  # X偏移值
    y_offset = 0  # Y偏移值
    z_offset = 0  # Z偏移值
    max_x = 0  # 里程真实最大X
    min_x = 0  # 里程真实最小X
    max_y = 0  # 真实最大Y
    min_y = 0  # 真实最小Y258
    max_z = 0  # 真实最大Z354
    min_z = 0  # 真实最小Z96
    rev = ''  # 预留

# 点数据
class LASPoint:
    point_source_id = 0  # 点源ID(点序号)
    x = 0  # X坐标
    y = 0  # Y坐标
    z = 0  # Z坐标
    intensity = 0  # 反射强度
    return_number = 0  # 反射号(回波)
    classification = ''  # 分类(不同设施分层管理)
    key_point = ''  # 是否关键点
    user_data = ''  # 用户可关联自定义数据
    color_id = 0  # 颜色ID(根据ID区分颜色)
    shape_id = ''  # 形状ID(根据ID区分形状)
    time = 0  # 距头文件Time的毫秒数
    curvature_radius = 0  # 曲率
    super = 0  # 超高
    longitude = 0  # 经度(预留)
    latitude = 0  # 纬度(预留)
    height = 0  # 高程(预留)

###读取repc头文件
def readLASHeader(filename) -> LASheader:
    '''
    读取repc文件头

    :return: repc文件头数据
    '''
    header = LASheader()
    f = open(filename, "rb")
    f.seek(0)
    # 读取文件头的350个字节
    s = f.read(350)
    # 使用unpack将读取到的字节解析为指定类型数据
    header.file_signature, header.file_id, header.major, header.minor, header.hardware_system, header.software_system, header.railway_name, header.railway_num, header.railway_direction, header.mile_direction, header.start_mileage, header.train_no, header.position, header.time, header.header_size, header.point_data_size, header.point_number, \
    header.x_scale_factor, header.y_scale_factor, header.z_scale_factor, header.x_offset, header.y_offset, header.z_offset, header.max_x, header.min_x, header.max_y, header.min_y, header.max_z, header.min_z, header.rev = struct.unpack(
        "<4sH2B30s30s30s10s2Bd20sBQHBL12d100s", s)
    f.close()
    return header


# Q3l2H3BHBL2h2LH
def readLASPoint(filename,index) -> LASPoint:
    '''
    读取一个点

    :param index: 点的索引，从1开始
    :return: 读取的点
    '''
    point = LASPoint()
    f = open(filename, "rb")
    f.seek(350 + 48 * (index - 1))
    # 读取一个数据点大小即43个字节的数据
    s = f.read(48)
    # 使用unpack将读取到的字节解析为指定类型数据
    # point.point_source_id, point.x, point.y, point.z, point.intensity, point.return_number, point.classification, point.key_point, point.user_data, point.color_id, point.shape_id, \
    # point.time, point.curvature_radius, point.super, point.longitude, point.latitude, point.height = struct.unpack(
    #     "<L3l2H5BL2h2LH", s)
    point.point_source_id, point.x, point.y, point.z, point.intensity, point.return_number, point.classification, point.key_point, point.user_data, point.color_id, point.shape_id, \
    point.time, point.curvature_radius, point.super, point.longitude, point.latitude, point.height = struct.unpack(
        "<Q3l2H3BHBL2h2LH", s)
    f.close()
    return point

def readLASPoints(filename,index, num) -> bytes:
    '''
    读取多个点的数据，但并不做解析

    :param index: 起始点的索引，从0开始
    :param num: 想要读取的点的个数
    :return: 未作解析的字节数据
    '''
    f = open(filename, "rb")
    f.seek(350 + 48 * index)
    s = f.read(num * 48)
    f.close()
    return s

def readRepcPoints(filename,index, num) -> list:  # 从index=0开始 num表示想读取的点的个数
    '''
    读取多个点的数据并解析为list形式返回

    :param index: 起始点的索引，从0开始
    :param num: 想要读取的点的个数
    :return: 读取到的多个点
    '''
    points = []
    f = open(filename, "rb")
    f.seek(350 + 48 * index)
    for i in range(num):
        point = LASPoint()
        # 读取一个数据点大小即43个字节的数据
        s = f.read(48)
        # 使用unpack将读取到的字节解析为指定类型数据
        # point.point_source_id, point.x, point.y, point.z, point.intensity, point.return_number, point.classification, point.key_point, point.user_data, point.color_id, point.shape_id, \
        # point.time, point.curvature_radius, point.super, point.longitude, point.latitude, point.height = struct.unpack(
        #     "<L3l2H5BL2h2LH", s)
        point.point_source_id, point.x, point.y, point.z, point.intensity, point.return_number, point.classification, point.key_point, point.user_data, point.color_id, point.shape_id, \
        point.time, point.curvature_radius, point.super, point.longitude, point.latitude, point.height = struct.unpack(
            "<Q3l2H3BHBL2h2LH", s)
        points.append(point)
    f.close()
    return points

def writeAllData(filename:str,points:List[LASPoint]):
    '''
    写入处理后的全部点数据

    Args:
        filename: 文件路径
        points: 全部点数据
    Returns: None

    '''
    f=open(filename,'rb+')
    f.seek(350,0)
    for i in points:
        s=struct.pack("<Q3l2H3BHBL2h2LH",i.point_source_id, i.x, i.y, i.z, i.intensity, i.return_number, i.classification, i.key_point, i.user_data, i.color_id, i.shape_id, \
        i.time, i.curvature_radius, i.super, i.longitude, i.latitude, i.height)
        f.write(s)
    f.close()

#读取所有原始点数据(恢复缩放后的数据)
def readAllData(filename):
    header = readLASHeader(filename)
    # point_number = header.point_number
    points = readRepcPoints(filename,0,header.point_number)
    data = []
    for i in range(len(points)):
        point_id = points[i].point_source_id
        x = points[i].x * header.x_scale_factor
        y = points[i].y * header.y_scale_factor
        z = points[i].z * header.z_scale_factor
        data.append([point_id,x,y,z])
    return points,data

def readAllDataAndLabel(filename):
    header = readLASHeader(filename)
    points = readRepcPoints(filename,0,header.point_number)
    data = []
    for i in range(len(points)):
        point_id = points[i].point_source_id
        x = points[i].x * header.x_scale_factor
        y = points[i].y * header.y_scale_factor
        z = points[i].z * header.z_scale_factor
        cls = points[i].classification
        data.append([point_id,x,y,z,cls])
    return data

def read_all_data_and_label(filename):
    header = readLASHeader(filename)
    points = readRepcPoints(filename,0,header.point_number)
    data = []
    for i in range(len(points)):
        point_id = points[i].point_source_id
        x = points[i].x * header.x_scale_factor
        y = points[i].y * header.y_scale_factor
        z = points[i].z * header.z_scale_factor
        cls = points[i].classification
        if cls == 11:
            cls = 1
        elif cls == 12:
            cls = 2
        elif cls == 13:
            cls = 3
        elif cls == 14:
            cls = 4
        elif cls == 15:
            cls = 5
        elif cls == 16:
            cls = 6
        elif cls == 17:
            cls = 7
        elif cls == 18:
            cls = 9
        elif cls == 19:
            cls = 8
        elif cls == 20:
            cls = 10
        data.append([point_id,x,y,z,cls])
    return data




# def readPoints(filename,index, num):  # 从index=0开始 num表示想读取的点的个数
#     '''
#     读取多个点的数据并解析为list形式返回
#
#     :param index: 起始点的索引，从0开始
#     :param num: 想要读取的点的个数
#     :return: 读取到的多个点
#     '''
#     points = np.zeros((num,4))
#     f = open(filename, "rb")
#     f.seek(350 + 48 * index)
#     for i in range(num):
#         s = f.read(20)
#         points[i, 0],points[i, 1],points[i, 2],points[i, 3] = struct.unpack(
#             "<Q3l", s)
#         f.seek(28,1)
#
#     f.close()
#     return points

# def readXYZ(filename):  # 从index=0开始 num表示想读取的点的个数
#     '''
#     读取多个点的数据并解析为list形式返回
#
#     :param index: 起始点的索引，从0开始
#     :param num: 想要读取的点的个数
#     :return: 读取到的多个点
#     '''
#     header = readLASHeader(filename)
#     num = header.point_number
#     points = np.zeros((num,4))
#     with open(filename, 'rb') as f:
#         f.seek(350,0)
#         barray = bytearray(f.read())
#         for i in range(num):
#             s = barray[i*48:i*48+20]
#             points[i, 0],points[i, 1],points[i, 2],points[i, 3] = struct.unpack(
#                 "<Q3l", s)
#
#     return points,barray



def readXYZ(filename):  # 从index=0开始 num表示想读取的点的个数
    '''
    读取多个点的数据并解析为list形式返回

    :param index: 起始点的索引，从0开始
    :param num: 想要读取的点的个数
    :return: 读取到的多个点
    '''
    header = readLASHeader(filename)
    num = header.point_number
    points = []
    with open(filename, 'rb') as f:
        f.seek(350,0)
        barray = bytearray(f.read())
        for i in range(num):
            s = barray[i*48:i*48+20]
            points.append(struct.unpack(
                "<Q3l", s))
    return points, barray, header.x_scale_factor,header.y_scale_factor,header.z_scale_factor

def readXYZCls(filename):  # 从index=0开始 num表示想读取的点的个数
    '''
    读取多个点的数据并解析为list形式返回

    :param index: 起始点的索引，从0开始
    :param num: 想要读取的点的个数
    :return: 读取到的多个点
    '''
    header = readLASHeader(filename)
    num = header.point_number
    points = []
    with open(filename, 'rb') as f:
        f.seek(350,0)
        barray = bytearray(f.read())
        for i in range(num):
            s = barray[i*48:i*48+25]
            st = struct.unpack(
                "<Q3l2HB", s)
            cls = st[6]
            if cls == 11:
                cls = 1
            elif cls == 12:
                cls = 2
            elif cls == 13:
                cls = 3
            elif cls == 14:
                cls = 4
            elif cls == 15:
                cls = 5
            elif cls == 16:
                cls = 6
            elif cls == 17:
                cls = 7
            elif cls == 18: #9
                cls = 9
            elif cls == 19: #8
                cls = 8
            elif cls == 20:
                cls = 10
            else:
                cls = 0
            points.append((st[0], st[1]*header.x_scale_factor, st[2]*header.y_scale_factor, st[3]*header.z_scale_factor, cls))
    return points

def writeLabel(filename:str,barray,labels):
    '''
    写入处理后的全部点数据

    Args:
        filename: 文件路径
        points: 全部点数据
    Returns: None

    '''
    with open(filename, 'rb+') as f:
        f.seek(350,0)
        lens = len(labels)
        for i in range(lens):
            cls = int(labels[i])
            change_cls = cls
            if cls == 1:
                change_cls = 11
            elif cls == 2:
                change_cls = 12
            elif cls == 3:
                change_cls = 13
            elif cls == 4:
                change_cls = 14
            elif cls == 5:
                change_cls = 15
            elif cls == 6:
                change_cls = 16
            elif cls == 7:
                change_cls = 17
            elif cls == 8:
                change_cls = 19
            elif cls == 9:
                change_cls = 18
            elif cls == 10:
                change_cls = 20
            new_cls = change_cls.to_bytes(1, byteorder='little', signed=False)
            before_cls = struct.unpack(
                "<B", barray[i*48+24:i*48+25])[0]
            if before_cls != 255:
                barray[i*48+24:i*48+25] = new_cls
        f.write(barray)



# def writeCls(filename:str,labels):
#     '''
#     写入处理后的全部点数据
#
#     Args:
#         filename: 文件路径
#         points: 全部点数据
#     Returns: None
#
#     '''
#     f=open(filename,'rb+')
#     f.seek(350+24,0)
#     len = labels.shape[0]
#     for i in range(len):
#         s=struct.pack("<B",labels[i])
#         f.write(s)
#         #跳过后续不需要写入的位置
#         f.seek(47,1)
#     f.close()

# def writeCls_2(filename:str,labels):
#     '''
#     写入处理后的全部点数据
#
#     Args:
#         filename: 文件路径
#         points: 全部点数据
#     Returns: None
#
#     '''
#     f=open(filename,'rb+')
#     f.seek(350,0)
#     start = time.time()
#     barray = bytearray(f.read())
#     end = time.time()
#     print("耗时: {:.2f}秒".format(end - start))
#     lens = len(labels)
#     for i in range(lens):
#         a = barray[i*48+25:i*48+26]
#         cls = struct.unpack(
#             "<B", a)
#         c = (1).to_bytes(1,byteorder='little',signed=False)
#         barray[i*48+25:i*48+26] = c
#         cc = struct.unpack(
#             "<B", c)
#         dd = cc = struct.unpack(
#             "<B", barray[i*48+25:i*48+26])
#
#
#     f.seek(0,0)
#     start = time.time()
#     f.write(barray)
#     end = time.time()
#     print("耗时: {:.2f}秒".format(end - start))
#     f.close()

# @jit()
# def test():
#     num = 0
#     for i in range(10000000):
#         num += i
#     print(num)


# filename = '测试数据/1.repc'
# start = time.time()
# points = readXYZ(filename)
# # test()
# end = time.time()
# print("耗时: {:.2f}秒".format(end - start))
# print('aaa')

# filename = '测试数据/5.repc'
# points,points_array = readXYZ(filename)
# labels = [5 for i in range(10000000)]
# writeLabel(filename,points_array,labels)

# points,data = readAllData(filename)
# print('aa')
# filename = '测试数据/aaa.repc'
# start = time.time()
# points,data = readAllData(filename)
# end = time.time()
# print("耗时: {:.2f}秒".format(end - start))
# start = time.time()
# writeAllData(filename,points)
# end = time.time()
# print("耗时: {:.2f}秒".format(end - start))


# filename = '测试数据/4.repc'
# points,barray1 = readPoints_2(filename,0,10000000)
# clss = [1 for i in range(10000000)]
# start = time.time()
# writeCls_3(filename,barray1,clss)
# end = time.time()
# print("耗时: {:.2f}秒".format(end - start))
# print('aaa')


# start = time.time()
# filename = '测试数据/aaa.repc'
# # data = readAllData(filename)
# header = readLASHeader(filename)
# data = readPoints(filename,0,header.point_number)
# end = time.time()
# print("耗时: {:.2f}秒".format(end - start))
# print('aaa')

# # start = time.time()



