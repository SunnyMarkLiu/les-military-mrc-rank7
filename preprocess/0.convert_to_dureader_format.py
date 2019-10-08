#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
将数据转换为 dureader 格式

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/3 16:21
"""
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import re
import json
import collections


# --------------- remove space ---------------
spaces = {'\x10', '\x7f', '\x9d', '\xad', '\x0a', '\xa0', '\x0d', '\u001d', '\u0007', '\u001f', '\u000f',
          '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u0009',
          '\u0010', '\u0011', '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018',
          '\f', '\n', '\r', '\t', '\v', '&#160;', '&nbsp;', '\\uDDEF', '\\uDDEE', '\\uDDE8', '\\uDDF3', '\\u0001',
          '\\uD83C', '\\uDDFA', '\\uD83C', '\\uDDF8', '\\uDDEA', '\\uDDF7', '\\uDDF5', '\\uDDF3',
          '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u1680', '\u180e',
          '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
          '\u2009', '\u200a', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}


def remove_unicode_space(text):
    for space in spaces:
        text = text.replace(space, '')
    text = re.sub('\s+', ' ', text)
    return text


# --------------- remove url ---------------
URL_REGEX1 = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
# 该正则会导致某些答案被清洗掉
# URL_REGEX2 = r'[^\u4e00-\u9fa5|[$-_@.&+]]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def remove_url_links(text):
    text = re.sub(URL_REGEX1, '', text)
    # text = re.sub(URL_REGEX2, '', text)
    return text


# --------------- remove html tag ---------------
html_pattern = re.compile(r'<.*?>')
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')


def remove_html_tag(text):
    text = re.sub('<{2,}', '', text)  # <<<
    # <> 中包含中文的
    text = re.sub('<；；', '', text)
    text = re.sub('([\u4e00-\u9fa5]+)>', '\g<1>', text)
    text = re.sub('<([\u4e00-\u9fa5]+)>', '\g<1>', text)
    tags = html_pattern.findall(text)

    new_tags = set()
    for tag in tags:
        match = zhPattern.search(tag)
        if not match:
            new_tags.add(tag)

    for html_tag in new_tags:
        text = text.replace(html_tag, '')

    return text


# --------------- remove duplacte chars ---------------
def clean_duplacte_chars(text):
    """
    去除很多重复的词和标点符号
    """
    reg = r'([^0-9IX]+)(\1){2,}'
    for i in range(6):
        temp = text
        text = re.sub(reg, lambda m: m.group(1), text)
        if len(text) == len(temp):
            break
    return text


remove_regx_map = collections.OrderedDict({
    r'\s+': ' ',
    r'<(\d+)>': '\g<1>',
    r'\^[A-Z]': '',  # '^G', '^H', '^E'去除
    r'(\!|\"|\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~)\1{1,}': '\g<1>',
    r'("""|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|　|、|〃|〈|〉|《|》|'
    r'「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏|﹑|﹔|·|！|？|｡|。)\1{1,}': '\g<1>',
    r'图\d+': '',
    r'\(记者[^\)]*\){1}': '',
    r'转自铁血社区': '',
    r'2019优选加盟：一点点奶茶，四季经营无淡季广告2019优选加盟：一点点奶茶，四季经营无淡季': '',
    r'０': '0', r'１': '1', r'２': '2', r'３': '3', r'４': '4',
    r'５': '4', r'６': '6', r'７': '7', r'８': '8', r'９': '9',
    r'．': '.'
})

def remove_by_regex(text):
    text = text.strip()
    for rgx in remove_regx_map:
        text = re.sub(rgx, remove_regx_map[rgx], text)
    return text


def clean_text(text, is_supporting_paragraph=False):
    # basic cleaning according to bad case sample
    text = text.replace('千米小时', '千米/小时')
    text = text.replace('. ,', '.')

    text = remove_unicode_space(text)
    text = remove_html_tag(text)

    if not is_supporting_paragraph:
        text = remove_url_links(text)

    text = clean_duplacte_chars(text)
    text = remove_by_regex(text)

    # 去除空格, 790个样本的答案中包含空格，占比 0.3 %
    text = text.replace(' ', '')
    return text


def clean_sample(sample):
    sample['question'] = clean_text(sample['question'])
    sample['keyword'] = clean_text(sample['keyword'])
    if 'answer' in sample:
        sample['answer'] = clean_text(sample['answer'], is_supporting_paragraph=True)
        sample['answer'] = sample['answer'].replace('@content1@content','@content1@@content') \
            .replace('@content2@content', '@content2@@content') \
            .replace('@content3@content', '@content3@@content') \
            .replace('@content4@content', '@content4@@content') \
            .replace('@content5@content', '@content5@@content')

    if 'supporting_paragraph' in sample:
        sample['supporting_paragraph'] = clean_text(sample['supporting_paragraph'], is_supporting_paragraph=True)
        sample['supporting_paragraph'] = sample['supporting_paragraph'].replace('@content1@content', '@content1@@content') \
                                            .replace('@content2@content', '@content2@@content') \
                                            .replace('@content3@content', '@content3@@content') \
                                            .replace('@content4@content', '@content4@@content') \
                                            .replace('@content5@content', '@content5@@content')

    del sample['keyword']

    for document in sample['documents']:
        document['paragraphs'] = [clean_text(para) for para in document['paragraphs']]
        del document['title']

        # 去除空段落和重复段落
        new_paras = []
        for para in document['paragraphs']:
            if para != '' and para not in new_paras:
                new_paras.append(para)
        document['paragraphs'] = new_paras

        document['content'] = ''.join(document['paragraphs'])
        del document['paragraphs']


ans_pattern = re.compile(r'@content\d@')


def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))


train_df = pd.read_csv('../input/original/train_round_0.csv', sep=',')
test_df = pd.read_csv('../input/original/test_data_r0.csv', sep=',')

train_file_out = open('../input/raw/train_round_0.json', 'w', encoding='utf8')
test_file_out = open('../input/raw/test_data_r0.json', 'w', encoding='utf8')

train_samples = []
for rid, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    sample = {'question_id': row['question_id'],
              'question': row['question'],
              'answer': row['answer'],
              'bridging_entity': None if row['bridging_entity'] == '无' else row['bridging_entity'],
              'keyword': row['keyword'],
              'supporting_paragraph': row['supporting_paragraph'],
              'documents': []}

    # 处理一些标注的 bad case
    if row['question_id'] == 'Q_07096kkkdj':
        sample['supporting_paragraph'] = "@content1@1944年年底美国陆军要求改进M26\"潘兴\"坦克，这便是后来的T32。@content1@content4@当今陆战之王，美国M1艾布拉姆斯系列主战坦克M60坦克M1坦克采用了乔巴姆复合装甲，乔巴姆装甲是一种多款材料组合的夹层装甲，在钢装甲中嵌入特种陶瓷。综合多款材料的防护优势，可以对破甲弹、反坦克导弹等弹药形成不同以往的防护效果。M1坦克的主炮为105毫米口径的M68线膛炮，炮弹基数55发，可以发射美军的M833贫铀穿甲弹。采用指挥仪式火控系统，火控系统包括激光测距仪、弹道计算机、炮长热成像系统、横风传感器和炮口校正传感器，使M1坦克得到了跨越式提升，并能在夜间环境遂行作战任务。M1坦克的动力系统为AGT-1500燃气轮机，采用带旋转减震器的高强度钢扭杆悬挂系统，这样以来，M1坦克就拥有了卓越的机动性，同时还具备良好的可靠性与安静性能。艾布拉姆斯坦克的炮塔还安装了车载机枪为了应对苏联的新型坦克发展@content4@"
    if row['question_id'] == 'Q_35449ydjjg':
        sample['supporting_paragraph'] = "@content1@Chariot或潜员运载工具是对意大利二战时著名的“猪”(Maiali)潜艇不断升级的结果。因为续航力的原因，Chariot通常由母舰如SWAT(浅水攻击型潜艇)、巡逻潜艇、直升机甚至于看上去无恶意的渔船运送它们到行动范围区内。Chariot是为了能够在敌方水域不被发现的航行，并把两名潜水突击队员运送到目标区(港口、钻油架、海岸设施等地方)然后在任务完成时能返回母舰而设计的。两名艇员骑跨于载具内，鱼雷状艇体。制造厂：CosmosSpa(意大利，里窝那)乘员编制：1人舰长：7米型宽：0.8米续航距离：50海里水上排水量：2吨潜航深度：100米武器装备一枚Mk32(230千克炸药)或两枚Mk415(105千克炸药)或一枚Mk41和五枚小型鱼雷，12枚Mk414水下爆破弹(每枚带有7千克炸药)或十枚Mk430水下爆破弹(每枚带有15千克炸药)。主要用户阿根廷埃及印度@content1@@content5@说起意大利军队，大家可能会想起那些离开意面就活不下去的少爷兵们。其实与陆军不同，意大利海军一直是欧洲一霸，特别是意大利的潜艇部队，在冷战时期给一直横行欧洲的苏联潜艇部队造成了不小的麻烦。@content5@"
    if row['question_id'] == 'Q_50970hhhdh':
        sample['supporting_paragraph'] = "@content3@SKS半自动步枪是由苏联枪械设计师西蒙诺夫（SergeiGavrilovichSimonov）在1945年设计的半自动步枪。SKS45是（SamozaryadniyKarabinsistemiSimonova，俄文：СамозарядныйкарабинсистемыСимонова，中文翻译是自动上膛的卡宾枪－西蒙诺夫系统，1945年）的缩写。SKS半自动步枪在苏联的第一线服役后很快的被AK-47取代。但其后它仍然在第二线服务了几十年。直到今天，仍然是俄罗斯、中国和多个前苏联国家的仪仗队使用的步枪。SKS半自动步枪布局与传统卡宾枪无异，使用木制枪托，没有手枪型式的握把。大多数版本都在枪管下配有一个可折叠的刺刀。SKS半自动步枪没有可拆卸的弹匣，只有固定弹仓，只能半自动射击。该枪的固定弹仓可以从容纳十发子弹的弹夹由机匣上方压入装填。弹仓亦可以由位于扳机护弓前的卡笋打开，弹仓中未发射的子弹可以快速除去。九九式短步枪全枪重：3.8千克@content3@@content5@10.第一名：AK-47突击步枪АК-47是由苏联枪械设计师米哈伊尔·季莫费耶维奇·卡拉什尼科夫（МихаилТимофеевичКалашников）设计的自动步枪。“AK”的意思是“АвтоматКалашникова”（“自动步枪”的首字母缩写）“47”的意思是“1947年产”。是苏联的第一代突击步枪。@content5@"
    if row['question_id'] == 'Q_14092hckjc':
        sample['supporting_paragraph'] = "@content2@斯普林菲尔德M1903步枪全枪重：3.9千克斯普林菲尔德M1903步枪是一种旋转后拉式枪机弹仓式手动步枪，1903年定型称为“0.30口径M1903步枪”，因其由斯普林菲尔德(Springfield)兵工厂研制而得名斯普林菲尔德步枪（Springfieldrifle），也有译成春田兵工厂而称为春田步枪，是美军在一战及二战的制式步枪。@content2@"

    if row['question_id'] == 'Q_35581ggncd':
        sample['bridging_entity'] = '鱼-8'
    if row['question_id'] == 'Q_15941hqhjj':
        sample['bridging_entity'] = '捷克ZB26轻机枪'
    if row['question_id'] == 'Q_37402ykkqk':
        sample['bridging_entity'] = '捷克ZB26轻机枪'
    if row['question_id'] == 'Q_06184qnsdj':
        sample['bridging_entity'] = 'AUG'
    if row['question_id'] == 'Q_49908chsss':
        sample['bridging_entity'] = '“悬崖”导弹'
    if row['question_id'] == 'Q_40999ghhgd':
        sample['bridging_entity'] = "A-10雷电攻击机"
    if row['question_id'] == 'Q_53387nyqcn':
        sample['bridging_entity'] = "竞技神号"
    if row['question_id'] == 'Q_27968cqhng':
        sample['bridging_entity'] = "66式152毫米加榴炮"

    ans_in_doc_ids = find_answer_in_docid(row['answer'])
    supported_doc_ids = find_answer_in_docid(row['supporting_paragraph'])

    for docid in range(1, 6):
        if docid in ans_in_doc_ids:
            is_selected = True
        else:
            is_selected = False

        # 注意：
        # 1. supporting_paragraph存在句号，所以在转成dureader时候按照句号进行切分句子存在缺陷！
        # 2. 观察数据发现按照双空格'  '划分段落
        paragraphs = [para.strip() for para in row['content{}'.format(docid)].split('  ')]
        paragraphs = [para for para in paragraphs if para != '']

        sample['documents'].append({
            'is_selected': is_selected,
            'title': row['title{}'.format(docid)],
            'paragraphs': paragraphs
        })

    clean_sample(sample)
    train_samples.append(json.dumps(sample, ensure_ascii=False) + '\n')

    if len(train_samples) % 1000 == 0:
        train_file_out.writelines(train_samples)
        train_file_out.flush()
        train_samples.clear()

if train_samples:
    train_file_out.writelines(train_samples)
    train_file_out.flush()
    train_file_out.close()

# --------------------------------- test ---------------------------------
test_samples = []
for rid, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    sample = {'question_id': row['question_id'],
              'question': row['question'],
              'keyword': row['keyword'],
              'documents': []}

    for docid in range(1, 6):
        # 注意：
        # 1. supporting_paragraph存在句号，所以在转成dureader时候按照句号进行切分句子存在缺陷！
        # 2. 观察数据发现按照双空格'  '划分段落
        paragraphs = [para.strip() for para in row['content{}'.format(docid)].split('  ')]
        paragraphs = [para for para in paragraphs if para != '']

        sample['documents'].append({
            'title': row['title{}'.format(docid)],
            'paragraphs': paragraphs
        })

    clean_sample(sample)
    test_samples.append(json.dumps(sample, ensure_ascii=False) + '\n')

    if len(test_samples) % 1000 == 0:
        test_file_out.writelines(test_samples)
        test_file_out.flush()
        test_samples.clear()

if test_samples:
    test_file_out.writelines(test_samples)
    test_file_out.flush()
    test_file_out.close()
