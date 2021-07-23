
"""
    本代码用于: 分析 MSCOCO 提供的数据
    创建时间: 2021 年 7 月 20 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 7 月 20 日
    具体步骤:
        step1: 分析 MSCOCO 的 data.json 文件
        step2: 分析 MSCOCO 的 data.h5 文件
        step3: 将 Ref 的 anns 框和正确框标注, 并保存为图片
        step4: refer 包的调用
"""
# -------------------- 导入必要的包 -------------------- #
# ----- 系统操作相关 ----- #
import time
import os
import pprint
import sys

# ----- 文件读取相关 ----- #
import h5py
import json 
import csv 

# ----- 数据处理相关 ----- #
import numpy as np
import copy

# ----- 图片处理相关 ----- #
from PIL import Image, ImageDraw, ImageFont 


# ----- refer 解析器 ----- #
sys.path.insert(0, 'tools_qxy/refer-python3')
from refer import REFER 
refer = REFER("/home/data/yjgroup/qxy/qxy/CM-Att-Erase/data/refcoco", "refcoco", "unc")

# -------------------- 设置常量参数 -------------------- #
# ----- 读取文件相关 ----- #
PATH_MSCOCO_DATA_JSON = "/home/data/yjgroup/qxy/qxy/CM-Att-Erase/cache/prepro/refcoco_unc/data.json"
PATH_MSCOCO_DATA_H5 = "/home/data/yjgroup/qxy/qxy/CM-Att-Erase/cache/prepro/refcoco_unc/data.h5"

PATH_MSCOCO_PIC = "/home/data/yjgroup/qxy/qxy/CM-Att-Erase/data/images"
split = ["train2014", "test2014", "val2014"]

# ----- 保存文件相关 ----- #
PATH_SAVE_WORD_TO_IX = "output_qxy/Data_Analyze/1-word_to_ix.csv"
PATH_SAVE_ATT_TO_IX = "output_qxy/Data_Analyze/2-att_to_ix.csv"
PATH_SAVE_ATT_TO_CNT = "output_qxy/Data_Analyze/3-att_to_cnt.csv"
PATH_SAVE_CAT_TO_IX = "output_qxy/Data_Analyze/4-cat_to_ix.csv"

PATH_SAVE_GROUND_TRUTH = "output_qxy/Data_Analyze/Ground_Truth"

# ----- 常量相关 ----- #
str_data_json = ['word_to_ix', 'label_length', 'sentences', 'images', 'att_to_ix', 
                 'refs', 'anns', 'att_to_cnt', 'cat_to_ix']

# -------------------- 函数实现 -------------------- #
# ----- 自动排序 + 保存排序后的字典为 csv ----- #
def Sorted_And_Save_Csv(path, data_raw, str):
    # ----- 排序实现 ----- #
    data = sorted(data_raw.items(), key=lambda x:x[1], reverse=True)

    # ----- 保存排序后的字典为 csv ----- #
    f = open(path, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow([str, "次数"])

    for key in data:
        csv_writer.writerow([key[0], key[1]])
    f.close()


# -------------------- 主函数运行 -------------------- #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # # ---------- step1: 分析 MSCOCO 的 data.json 文件 ---------- #
    # # ----- 读取 JSON 文件 ----- #
    # with open(PATH_MSCOCO_DATA_JSON, 'r') as f:
    #     data = json.load(f)
    #     # print(data.keys())

    #     # ----- word_to_ix 的输出 ----- #
    #     print("==================================================")
    #     print("word_to_ix")
    #     # print(data["word_to_ix"])
    #     dict_word_to_ix = data["word_to_ix"]
    #     print("word_to_ix 的个数为: ", len(data["word_to_ix"]))
    #     print("word_to_ix: 将 Word 转换为相应的 index")

    #     # ----- label_length 的输出 ----- #
    #     print("==================================================")
    #     print("label_length")
    #     print(data["label_length"])

    #     # ----- sentences 的输出 ----- #
    #     print("==================================================")
    #     print("sentences")
    #     print("sentences 的数量为: ", len(data["sentences"]))
    #     for temp in data["sentences"]:
    #         pprint.pprint(temp, indent=4)
    #         break
    #     print("0: ", data["sentences"][0])
    #     print("1: ", data["sentences"][1])
    #     print("2: ", data["sentences"][2])

    #     # ----- images 的输出 ----- #
    #     print("==================================================")
    #     print("images")
    #     print("images 的数量为: ", len(data["images"]))
    #     for temp in data["images"]:
    #         pprint.pprint(temp, indent=4)
    #         break

    #     # ----- att_to_ix 的输出 ----- #
    #     print("==================================================")
    #     print("att_to_ix")
    #     dict_att_to_ix = data["att_to_ix"]
    #     # print(data["att_to_ix"])
    #     print("att_to_ix 的个数为: ", len(dict_att_to_ix))
    #     print("att_to_ix: 将 Attribute 转换为 index")

    #     # ----- refs 的输出 ----- #
    #     print("==================================================")
    #     print("refs")
    #     print("refs 的数量为: ", len(data["refs"]))
    #     for temp in data["refs"]:
    #         pprint.pprint(temp, indent=4)
    #         break

    #     # ----- anns 的输出 ----- #
    #     print("==================================================")
    #     print("anns")     
    #     print("anns 的数量为: ", len(data["anns"]))   
    #     for temp in data["anns"]:
    #         pprint.pprint(temp, indent=4)
    #         break


    #     # ----- att_to_cnt 的输出 ----- #
    #     print("==================================================")
    #     print("att_to_cnt")
    #     # print(data["att_to_cnt"])
    #     dict_att_to_cnt = data["att_to_cnt"]
    #     print("att_to_cnt 的个数为: ", len(data["att_to_cnt"]))
    #     print("att_to_cnt: 统计 Attribute 的个数")

    #     # ----- cat_to_ix 的输出 ----- #
    #     print("==================================================")
    #     print("cat_to_ix")
    #     # print(data["cat_to_ix"])
    #     dict_cat_to_ix = data["cat_to_ix"]
    #     print("cat_to_ix 的个数为: ", len(dict_cat_to_ix))
    #     print("cat_to_ix: 将 category 转换为 index")


    # # ----- 保存文件 ----- #
    # Sorted_And_Save_Csv(PATH_SAVE_WORD_TO_IX, dict_word_to_ix, "word_to_ix")
    # Sorted_And_Save_Csv(PATH_SAVE_ATT_TO_IX, dict_att_to_ix, "att_to_ix")
    # Sorted_And_Save_Csv(PATH_SAVE_ATT_TO_CNT, dict_att_to_cnt, "att_to_cnt")
    # Sorted_And_Save_Csv(PATH_SAVE_CAT_TO_IX, dict_cat_to_ix, "cat_to_ix")


    # # ---------- step2: 分析 MSCOCO 的 data.h5 文件 ---------- #
    # # ----- 读取 data.h5 文件 ----- #
    # data = h5py.File(PATH_MSCOCO_DATA_H5, 'r')
    # print(data.keys())
    # print(data["labels"])
    # print(data["labels"][0])
    
    # data.close()

    # ----- step3: 将 Ref 的 anns 框和正确框标注, 并保存为图片 ----- #
    # ----- 用于保存常量 ----- #
    dict_pic_id = {}
    dict_ref_id = {}
    dict_sent_id = {}
    dict_anns_id = {}

    # ----- 读取 data.json 获得基本信息 ----- #
    with open(PATH_MSCOCO_DATA_JSON, 'r') as f:
        data = json.load(f)
        data_pic = data["images"]
        data_ref = data["refs"]
        data_sent = data["sentences"]
        data_anns = data["anns"]

        # ----- 遍历 images ----- #
        for temp in data_pic:
            pic_file_name = temp["file_name"]
            pic_ann_ids = temp["ann_ids"]
            pic_ref_ids = temp["ref_ids"]
            
            dict_pic_id[pic_file_name] = {}
            dict_pic_id[pic_file_name]["ann_ids"] = pic_ann_ids
            dict_pic_id[pic_file_name]["ref_ids"] = pic_ref_ids


        # ----- 遍历 refs ----- #
        for temp in data_ref:
            ref_ann_id = temp["ann_id"]
            ref_att_wds = temp["att_wds"]
            ref_box = temp["box"]
            ref_id = temp["ref_id"]
            ref_sent_ids = temp["sent_ids"]
            ref_split = temp["split"]

            dict_ref_id[ref_id] = {}
            dict_ref_id[ref_id]["ann_id"] = ref_ann_id
            dict_ref_id[ref_id]["att_wds"] = ref_att_wds
            dict_ref_id[ref_id]["box"] = ref_box
            dict_ref_id[ref_id]["sent_ids"] = ref_sent_ids
            dict_ref_id[ref_id]["split"] = ref_split


        # ----- 遍历 sent ----- #
        for temp in data_sent:
            sent_id = temp["sent_id"]
            sent_token = temp["tokens"]

            str_token = ""
            for temp_token in sent_token:
                str_token += temp_token 
                str_token += " "

            dict_sent_id[sent_id] = str_token
            

        # ----- 遍历 anns ----- #
        for temp in data_anns:
            anns_id = temp["ann_id"]
            anns_box = temp["box"]
            dict_anns_id[anns_id] = anns_box


    # ----- 开始画图 ----- #
    i = 0
    # 遍历所有的图片
    for temp_img_id in dict_pic_id:
        # ----- 获取基本信息 ----- # 
        temp_img_ann = dict_pic_id[temp_img_id]["ann_ids"]
        temp_img_ref = dict_pic_id[temp_img_id]["ref_ids"]

        if 'train' in temp_img_id:
            # 遍历所有的 ref 
            j = 0
            for temp_ref in temp_img_ref:
                temp_img = Image.open(os.path.join(PATH_MSCOCO_PIC, split[0], temp_img_id))

                # 获取图片的大小信息
                row, col = temp_img.size 
                print(row, col)
                blank_height = 200

                # 建立保存图片的信息
                save_image = Image.new("RGBA", (row, col+blank_height))

                # 获取基本信息
                temp_ref_ann = dict_ref_id[temp_ref]["ann_id"]
                temp_ref_sent_id = dict_ref_id[temp_ref]["sent_ids"]

                # 去除当前框的 ann
                temp_ref_ann_list = copy.deepcopy(temp_img_ann)
                temp_ref_ann_list.remove(temp_ref_ann)
                # print("temp_img_ann: ", temp_img_ann)
                # print(temp_ref_ann_list)

                img_draw = ImageDraw.ImageDraw(temp_img)

                # ----- 画出其他矩形 ----- #
                for temp in temp_ref_ann_list:
                    box_ref = dict_anns_id[temp]
                    x0 = box_ref[0]
                    y0 = box_ref[1]
                    x1 = box_ref[2]
                    y1 = box_ref[3]
                    img_draw.rectangle((x0, y0, x1, y1), fill=None, outline="yellow", width=3)

                # ----- 画矩形 ----- #
                box_ref = dict_anns_id[temp_ref_ann]
                print("temp_ref: ", temp_ref)
                print(refer.getRefBox(temp_ref_ann))
                # print("temp_ref_ann: ", temp_ref_ann)
                # print(dict_ref_id[temp_ref]["box"])
                # print("att_wd: ", dict_ref_id[temp_ref]["att_wds"])
                # print("sent_ids: ", dict_sent_id[dict_ref_id[temp_ref]["sent_ids"][0]])
                x0 = box_ref[0]
                y0 = box_ref[1]
                x1 = box_ref[2]
                y1 = box_ref[3]
                img_draw.rectangle((x0, y0, x1, y1), fill=None, outline="red", width=5)


                # ----- 写字体 ----- #
                loc = (0, blank_height)
                save_image.paste(temp_img, loc)
                background = Image.new("RGB", (row, blank_height), (255,255,255))
                loc = (0, 0)
                save_image.paste(background, loc)
                save_image_draw = ImageDraw.Draw(save_image)
                font_temp = ImageFont.truetype("SimHei.ttf", 30) # 字体大小
                # 遍历句子
                k = 0
                for temp_sent_id in dict_ref_id[temp_ref]["sent_ids"]:
                    temp_sent = dict_sent_id[dict_ref_id[temp_ref]["sent_ids"][k]]    
                    save_image_draw.text((0, k*35), temp_sent, fill="blue", font=font_temp)
                    k += 1

                str_save_img = os.path.join(PATH_SAVE_GROUND_TRUTH, str(temp_ref)+".png")
                # temp_img.save(str_save_img)
                save_image.save(str_save_img)


                # 用于遍历
                j += 1

        elif 'test' in temp_img_id:
            print("test")
        elif 'val' in temp_img_id:
            print("val")
        else:
            print(temp_img_id)

        i += 1
        if i > 2:
            break

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))

    
    print("程序已结束 ！")