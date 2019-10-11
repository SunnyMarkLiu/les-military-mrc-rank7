import re


class LesQuestionTypeHandler(object):
    def __init__(self):
        pattern_who = re.compile('什么人|谁|哪位')
        pattern_how = re.compile('怎样|如何|怎么')
        pattern_time = re.compile('多久|多长时间|何时|什么时候|[那|哪][\s\S]{0,3}(秒|分钟|小时|天|星期|月|年|世纪|期)|时间$')
        pattern_num = re.compile(
            '多少|多高|多远|多重|多大|多长|多厚|厚度|多深|几|费用|比例|通过率|额度|手续费|速度|电话|金额|价格|税费|分数线|概率|尺寸|重量|排水量|寿命|范围')
        pattern_where = re.compile('哪里|在哪|什么[\s\S]*?地方|到哪|去哪|从哪|何处')
        pattern_why = re.compile('动因|为什么|为啥|什么原因|由来|原因|为了？|意义|为何|有何|作用|用于？|意旨|因何|目的？|在于|旨在')
        pattern_what = re.compile(
            '叫做|叫？|是啥|什么|[哪|那][一|两|二|三|四|五|六|七|八|九|十|几|三大]?[0-9]?[类|片|系列|航母|枚|级|只|把|方|支|个|些|种|集|国|家|艘|款|辆|架|号|代|场|项|大|处|次|型号|部分|方面]|何种|[是|为][？|\?]|参数')
        self.pattern_list = [pattern_who, pattern_how, pattern_time, pattern_num, pattern_where, pattern_why, pattern_what]
        self.pattern_names = ['who', 'how', 'time', 'num', 'where', 'why', 'what']
        assert len(self.pattern_list) == len(self.pattern_names)
        self.label_distrib = [0] * (len(self.pattern_list) + 1)

    def get_classify_label(self, text):
        """
        返回text对应的label_id以及label_name
        """
        for pid, (pattern, name) in enumerate(zip(self.pattern_list, self.pattern_names)):
            if re.findall(pattern, text):
                self.label_distrib[pid] += 1
                return pid, name
        # 未匹配
        self.label_distrib[-1] += 1
        return len(self.pattern_list), 'unk'

    def show_distrib(self):
        """
        打印各个标签的分布情况
        """
        all_cnt = sum(self.label_distrib)
        if all_cnt == 0:
            print('all fine classify label is zero!')
            return
        for name, cnt in zip(self.pattern_names + ['unk'], self.label_distrib):
            print('{}: {} -> {}'.format(name, cnt, cnt / all_cnt))
