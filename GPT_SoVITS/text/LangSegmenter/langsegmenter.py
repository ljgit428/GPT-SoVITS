import logging
import re

# jieba静音
import jieba
jieba.setLogLevel(logging.CRITICAL)

# 更改fast_langdetect大模型位置
from pathlib import Path
import fast_langdetect
fast_langdetect.infer._default_detector = fast_langdetect.infer.LangDetector(fast_langdetect.infer.LangDetectConfig(cache_dir=Path(__file__).parent.parent.parent / "pretrained_models" / "fast_langdetect"))


from split_lang import LangSplitter


def full_en(text):
    pattern = r'^(?=.*[A-Za-z])[A-Za-z0-9\s\u0020-\u007E\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]+$'
    return bool(re.match(pattern, text))


def full_cjk(text):
    # 来自wiki
    cjk_ranges = [
        (0x4E00, 0x9FFF),        # CJK Unified Ideographs
        (0x3400, 0x4DB5),        # CJK Extension A
        (0x20000, 0x2A6DD),      # CJK Extension B
        (0x2A700, 0x2B73F),      # CJK Extension C
        (0x2B740, 0x2B81F),      # CJK Extension D
        (0x2B820, 0x2CEAF),      # CJK Extension E
        (0x2CEB0, 0x2EBEF),      # CJK Extension F
        (0x30000, 0x3134A),      # CJK Extension G
        (0x31350, 0x323AF),      # CJK Extension H
        (0x2EBF0, 0x2EE5D),      # CJK Extension H
    ]

    pattern = r'[0-9、-〜。！？.!?… /]+$'

    cjk_text = ""
    for char in text:
        code_point = ord(char)
        in_cjk = any(start <= code_point <= end for start, end in cjk_ranges)
        if in_cjk or re.match(pattern, char):
            cjk_text += char
    return cjk_text


def split_jako(tag_lang,item):
    if tag_lang == "ja":
        pattern = r"([\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]+(?:[0-9、-〜。！？.!?… ]+[\u3041-\u3096\u3099\u309A\u30A1-\u30FA\u30FC]*)*)"
    else:
        pattern = r"([\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]+(?:[0-9、-〜。！？.!?… ]+[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]*)*)"

    lang_list: list[dict] = []
    tag = 0
    for match in re.finditer(pattern, item['text']):
        if match.start() > tag:
            lang_list.append({'lang':item['lang'],'text':item['text'][tag:match.start()]})

        tag = match.end()
        lang_list.append({'lang':tag_lang,'text':item['text'][match.start():match.end()]})

    if tag < len(item['text']):
        lang_list.append({'lang':item['lang'],'text':item['text'][tag:len(item['text'])]})

    return lang_list


def merge_lang(lang_list, item):
    if lang_list and item['lang'] == lang_list[-1]['lang']:
        lang_list[-1]['text'] += item['text']
    else:
        lang_list.append(item)
    return lang_list


class LangSegmenter():
    # 默认过滤器, 基于gsv目前四种语言
    DEFAULT_LANG_MAP = {
        "zh": "zh",
        "yue": "zh",  # 粤语
        "wuu": "zh",  # 吴语
        "zh-cn": "zh",
        "zh-tw": "x", # 繁体设置为x
        "ko": "ko",
        "ja": "ja",
        "en": "en",
    }

    def detect_language(text: str) -> str:
        """简化版语言检测（需根据实际需求完善）"""
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):  # 平假名/片假名
            return 'ja'
        if re.search(r'[\uAC00-\uD7A3]', text):  # 韩文字符
            return 'ko'
        if re.search(r'[\u4E00-\u9FFF]', text):  # 中文汉字
            return 'zh'
        return 'en'  # 默认英语

    # def getTexts(original_text: str) -> list[dict]:
    #   """将带标签文本转换为语言片段列表
      
    #   :param original_text: 包含语言标签的原始文本，示例: "<ja>こんにちは</ja><zh>你好</zh>"
    #   :return: 结构化语言列表，示例: [{'lang':'ja', 'text':'こんにちは'}, {'lang':'zh', 'text':'你好'}]
    #   """
    #   print('GETTING TEXTS')
    #   # 第一步：标签提取与文本清洗
    #   tag_pattern = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
    #   matches = tag_pattern.finditer(original_text)
    #   clean_text = tag_pattern.sub(r'\2', original_text)  # 删除标签后的纯净文本

    #   # 第二步：构建片段映射表
    #   segment_map = []
    #   for match in matches:
    #       lang = match.group(1).lower()
    #       tagged_text = match.group(2)
    #       start = match.start(2)  # 标签内容起始位置
    #       end = match.end(2)      # 标签内容结束位置
    #       segment_map.append({
    #           'lang': lang,
    #           'start': start,
    #           'end': end,
    #           'length': end - start
    #       })

    #   # 第三步：内容对齐检测
    #   pointer = 0
    #   lang_list = []
    #   for seg in sorted(segment_map, key=lambda x: x['start']):
    #       # 处理标签前未标注的文本
    #       if seg['start'] > pointer:
    #           unmarked_text = clean_text[pointer:seg['start']]
    #           lang_list.append({
    #               'lang': LangSegmenter.detect_language(unmarked_text),  # 需要实现语言检测函数
    #               'text': unmarked_text
    #           })
          
    #       # 添加已标注片段
    #       lang_list.append({
    #           'lang': seg['lang'],
    #           'text': clean_text[seg['start']:seg['end']]
    #       })
    #       pointer = seg['end']

    #   # 处理末尾未标注文本
    #   if pointer < len(clean_text):
    #       lang_list.append({
    #           'lang': LangSegmenter.detect_language(clean_text[pointer:]),
    #           'text': clean_text[pointer:]
    #       })

    #   return lang_list

    def strict_tag_split(text: str) -> list[dict]:
        """严格按XML标签分割文本，每个标签区段作为独立单元"""
        # 强化正则匹配，确保捕获完整标签
        pattern = re.compile(r'<(\w+?)>(.*?)</\1>', re.DOTALL)
        
        result = []
        last_pos = 0
        
        # 遍历所有标签匹配项
        for match in pattern.finditer(text):
            # start = match.start()
            # end = match.end()
            
            # # 前导无标签文本
            # if start > last_pos:
            #     result.append({
            #         'lang': LangSegmenter.detect_language(text[last_pos:start]),
            #         'text': text[last_pos:start]
            #     })
            
            # 处理当前标签内容
            lang = match.group(1).lower()
            content = match.group(2)
            result.append({
                'lang': lang,
                'text': content
            })
            
            # last_pos = end
        
        # 处理末尾文本
        # if last_pos < len(text):
        #     result.append({
        #         'lang': LangSegmenter.detect_language(text[last_pos:]),
        #         'text': text[last_pos:]
        #     })
        
        return result

    def contains_tags(text: str) -> bool:
      """高效检测文本是否包含标签结构 (包含防御性检测)"""
      # 预编译正则增强性能
      tag_detect = re.compile(r'<\/?[a-zA-Z][^>]*>', re.UNICODE)
      # 中文语境常见错误标签模式 (例如：<zh>, <ja>)
      zh_tag_pattern = re.compile(r'<(zh|ja|ko)\b', re.IGNORECASE)
      
      return bool(tag_detect.search(text)) or bool(zh_tag_pattern.search(text))
    
    
    def getTexts(text):
      if LangSegmenter.contains_tags(text):
        # 启用严格标签模式
        return LangSegmenter.strict_tag_split(text)

      if '|' in text:
        segments = text.split('|')
        results = []
        for seg in segments:
          # 第二级：标签检测
          results.extend(LangSegmenter.getTexts(seg))  
        return results

      lang_splitter = LangSplitter(lang_map=LangSegmenter.DEFAULT_LANG_MAP)
      substr = lang_splitter.split_by_lang(text=text)

      lang_list: list[dict] = []

      for _, item in enumerate(substr):
          dict_item = {'lang':item.lang,'text':item.text}

          # 处理短英文被识别为其他语言的问题
          if full_en(dict_item['text']):  
              dict_item['lang'] = 'en'
              lang_list = merge_lang(lang_list,dict_item)
              continue

          # 处理非日语夹日文的问题(不包含CJK)
          ja_list: list[dict] = []
          if dict_item['lang'] != 'ja':
              ja_list = split_jako('ja',dict_item)

          if not ja_list:
              ja_list.append(dict_item)

          # 处理非韩语夹韩语的问题(不包含CJK)
          ko_list: list[dict] = []
          temp_list: list[dict] = []
          for _, ko_item in enumerate(ja_list):
              if ko_item["lang"] != 'ko':
                  ko_list = split_jako('ko',ko_item)

              if ko_list:
                  temp_list.extend(ko_list)
              else:
                  temp_list.append(ko_item)

          # 未存在非日韩文夹日韩文
          if len(temp_list) == 1:
              # 未知语言检查是否为CJK
              if dict_item['lang'] == 'x':
                  cjk_text = full_cjk(dict_item['text'])
                  if cjk_text:
                      dict_item = {'lang':'zh','text':cjk_text}
                      lang_list = merge_lang(lang_list,dict_item)
                  else:
                      lang_list = merge_lang(lang_list,dict_item)
                  continue
              else:
                  lang_list = merge_lang(lang_list,dict_item)
                  continue

          # 存在非日韩文夹日韩文
          for _, temp_item in enumerate(temp_list):
              # 未知语言检查是否为CJK
              if temp_item['lang'] == 'x':
                  cjk_text = full_cjk(dict_item['text'])
                  if cjk_text:
                      dict_item = {'lang':'zh','text':cjk_text}
                      lang_list = merge_lang(lang_list,dict_item)
                  else:
                      lang_list = merge_lang(lang_list,dict_item)
              else:
                  lang_list = merge_lang(lang_list,temp_item)

      temp_list = lang_list
      lang_list = []
      for _, temp_item in enumerate(temp_list):
          if temp_item['lang'] == 'x':
              if lang_list:
                  temp_item['lang'] = lang_list[-1]['lang']
              elif len(temp_list) > 1:
                  temp_item['lang'] = temp_list[1]['lang']
              else:
                  temp_item['lang'] = 'zh'

          lang_list = merge_lang(lang_list,temp_item)

      return lang_list
    


if __name__ == "__main__":
    text = "MyGO?,你也喜欢まいご吗？"
    print(LangSegmenter.getTexts(text))

    text = "ねえ、知ってる？最近、僕は天文学を勉強してるんだ。君の瞳が星空みたいにキラキラしてるからさ。"
    print(LangSegmenter.getTexts(text))
