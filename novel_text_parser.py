import re
import numpy as np
import torch

class NovelTextParser:
    """
    解析小说文本，将其结构化为不同Character的对话和旁白
    """
    
    def __init__(self):
        # Character映射表 {Character标识: CharacterID}
        self.role_map = {}
        self.next_role_id = 1
        # 说话动词库（可扩展）
        self.speech_verbs = {"说", "问道", "喊道", "低声", "问", "答", "笑", "叹", "回应", "回答", "响起", "笑道", "道", 
                           "说道", "叫道", "念道", "解释", "回道", "吼道", "喃喃", "感叹", "插嘴", "呢喃", "咆哮", 
                           "呐喊", "哭诉", "嘟囔", "嘀咕", "抱怨", "打断", "反驳", "辩解", "质问", "追问", "附和", 
                           "应和", "赞同", "反对", "嗤笑", "冷笑", "大笑", "苦笑", "微笑", "轻声", "高声", "尖叫", 
                           "嚷嚷", "嚎叫", "呻吟", "哀叹", "感慨", "嘱咐", "命令", "告诫", "劝告", "建议", "提醒", 
                           "强调", "补充", "继续", "沉思", "自语", "喊", "讲", "谈", "评论", "议论", "宣布", "声明", 
                           "陈述", "表示", "暗示", "指出", "分析", "总结", "回忆", "思考", "开口", "呼唤", "祈求", 
                           "哀求", "恳求", "央求", "嘲讽", "讥讽", "挖苦", "调侃", "戏谑", "调笑", "戏弄", "吐槽"}
        
        # 常见非人名词汇，用于过滤
        self.non_character_words = {"这个", "那个", "他", "她", "它", "你", "我", "那", "这", "其", "某", 
                                 "谁", "哪", "是", "有", "没", "就", "可", "能", "会", "个", "的", "了",
                                 "着", "被", "让", "给"}
        
        # 角色标签模式
        self.role_tag_pattern = re.compile(r'<(Narrator|Character\d+)>')
        
    def _is_preformatted(self, text):
        """检测文本是否已经是预格式化的（已包含角色标签）
        
        Args:
            text: 输入文本
            
        Returns:
            bool: 是否预格式化
        """
        # 查找是否至少有一个角色标签
        tags_found = len(re.findall(self.role_tag_pattern, text)) > 0
        return tags_found
        
    def parse_text(self, text):
        """
        解析文本，将其结构化为不同Character的对话和旁白
        
        Args:
            text: 输入的小说文本
            
        Returns:
            structured_text: 结构化后的文本 (包含Character标签)
        """
        # 检测是否已经是预格式化的文本
        if self._is_preformatted(text):
            print("[Novel Text Parser] Detected pre-formatted text with role tags, preserving as-is")
            # 将预格式化文本转换为结构化格式
            segments = []
            current_idx = 0
            
            # 遍历所有标签匹配
            for match in re.finditer(self.role_tag_pattern, text):
                role = match.group(1)  # 角色名（Narrator或CharacterX）
                start_idx = match.end()  # 标签后的开始索引
                
                # 查找下一个标签的开始位置
                next_tag = re.search(self.role_tag_pattern, text[start_idx:])
                if next_tag:
                    end_idx = start_idx + next_tag.start()
                else:
                    end_idx = len(text)
                    
                # 提取文本内容
                content = text[start_idx:end_idx]
                segments.append({"type": role, "text": content})
                
            return segments
        
        # 预处理：按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        structured = []
        
        for para in paragraphs:
            # 1. 先检测引号和对话模式（更完善的引号判断）
            # 模式1："人物说道：“对话”" 或 "人物道：“对话”"
            dialogue_match = re.search(r'(.+?)(?:\s*[\u8bf4|\u9053].+?[\uff1a|"])["|\u201c](.+?)["|\u201d]', para)
            
            if dialogue_match:
                context, dialogue = dialogue_match.groups()
                # 从上下文中识别角色
                role_id = self._identify_speaker(context)
                structured.append({"type": "Narrator", "text": context})
                structured.append({"type": role_id, "text": dialogue.strip()})
                
            # 模式2：纯引号对话“对话”
            elif quote_match := re.search(r'["|\u201c](.+?)["|\u201d]', para):
                dialogue = quote_match.group(1)
                # 提取引号前后的上下文
                pre_context = para[:quote_match.start()]
                post_context = para[quote_match.end():]
                
                # 如果引号后有说话者描述，优先使用
                if post_context and any(verb in post_context for verb in self.speech_verbs):
                    role_id = self._identify_speaker(post_context)
                    structured.append({"type": role_id, "text": dialogue.strip()})
                    structured.append({"type": "Narrator", "text": post_context.strip()})
                # 如果引号前有说话者描述和动词
                elif pre_context and any(verb in pre_context for verb in self.speech_verbs):
                    role_id = self._identify_speaker(pre_context)
                    structured.append({"type": "Narrator", "text": pre_context.strip()})
                    structured.append({"type": role_id, "text": dialogue.strip()})
                # 如果无法确定说话者，将整段文本当作旁白
                else:
                    structured.append({"type": "Narrator", "text": para})
            # 模式3：纯叙述文本
            else:
                structured.append({"type": "Narrator", "text": para})
                
        return structured
    
    def format_structured_text(self, structured):
        """
        将结构化的文本格式化为带标签的文本
        
        Args:
            structured: 结构化的文本列表
            
        Returns:
            formatted_text: 格式化后的带标签文本
        """
        formatted = []
        for item in structured:
            text_type = item["type"]
            text = item["text"]
            if text_type == "Narrator":
                formatted.append(f"<Narrator>{text}")
            else:
                # 确保CharacterID格式为 "Character1", "Character2" 等
                if text_type.startswith("Character") and text_type[2:].isdigit():
                    formatted.append(f"<{text_type}>{text}")
                else:
                    # 默认情况下，尝试将Character映射到Character1-5
                    role_num = int(text_type[2:]) if text_type[2:].isdigit() else 1
                    if 1 <= role_num <= 5:
                        formatted.append(f"<Character{role_num}>{text}")
                    else:
                        formatted.append(f"<Narrator>{text}")
        
        return "".join(formatted)
    
    def _is_direct_speech(self, text):
        # 检测引导词或直接引号
        quotes = any(c in text for c in ['"', '"', '"', "'", "'", "'"])
        has_verb = any(verb in text for verb in self.speech_verbs)
        return quotes or has_verb
    
    def _extract_dialogue(self, text):
        # 提取引号内内容
        if match := re.search(r'[""](.+?)[""]', text):
            dialogue = match.group(1)
            # 从上下文识别Character
            context = text.replace(f'"{dialogue}"', '').replace(f'"{dialogue}"', '')
            return self._identify_role(context), dialogue
        return self._identify_role(text), text
    
    def _identify_speaker(self, context):
        """Enhanced speaker identification from dialogue context
        
        Args:
            context: surrounding text context
            
        Returns:
            role_id: the identified speaker's role ID
        """
        # 1. 检测已知Character名
        for name, role_id in self.role_map.items():
            if name in context:
                return role_id
                
        # 2. 尝试匹配可能的中文人名模式
        # 匹配姓名常见的 2-3 字的名字，及姓名前后带有说话动词的
        name_match = re.search(r'([一-龥]{2,3})(?:[^\n\r]{0,10}[\u8bf4\u9053])', context)
        if name_match:
            new_role = name_match.group(1).strip()
            # 过滤掉常见的非人名词汇
            if new_role and new_role not in self.non_character_words:
                role_id = f"Character{min(self.next_role_id, 5)}"  # 限制最多到Character5
                print(f"[Novel Text Parser] Identified new character: {new_role} as {role_id}")
                self.role_map[new_role] = role_id
                self.next_role_id = min(self.next_role_id + 1, 6)  # 最多到Character5
                return role_id
        
        # 3. 更广泛的匹配 - 寻找符合中文人名特征的词语
        characters = re.findall(r'[一-龥]{2,3}', context)
        for char in characters:
            # 过滤掉常见的非人名词汇
            if len(char) >= 2 and char not in self.non_character_words:
                role_id = f"Character{min(self.next_role_id, 5)}"
                print(f"[Novel Text Parser] Inferred character name: {char} as {role_id}")
                self.role_map[char] = role_id
                self.next_role_id = min(self.next_role_id + 1, 6)
                return role_id
                
        # 4. 默认处理
        return "Narrator"

    def _identify_role(self, context):
        # 向后兼容保留的方法，调用新的方法
        return self._identify_speaker(context)


# ComfyUI节点：小说文本结构化节点
class NovelTextStructureNode:
    """
    ComfyUI的小说文本结构化节点，用于将小说文本结构化为不同Character的对话和旁白
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "novel_text": ("STRING", {"multiline": True, "default": 'Novel text example.\nLin Wei said, "Hello there."\nSu Qing replied, "Long time no see."\n'}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("structured_text",)
    FUNCTION = "structure_novel_text"
    CATEGORY = "text/novels"
    
    def __init__(self):
        self.parser = NovelTextParser()
        print("[Novel Text Structure Node] Initialization completed")
    
    def structure_novel_text(self, novel_text):
        """
        将小说文本结构化为不同Character的对话和旁白
        
        Args:
            novel_text: 输入的小说文本
            
        Returns:
            structured_text: 结构化后的文本
        """
        try:
            print(f"[Novel Text Structure] Processing novel text, length: {len(novel_text)}")
            
            # 解析文本
            structured = self.parser.parse_text(novel_text)
            
            # Character统计
            role_stats = {}
            for item in structured:
                role = item["type"]
                if role not in role_stats:
                    role_stats[role] = 0
                role_stats[role] += 1
                
            print(f"[Novel Text Structure] 解析完成，识别到Character统计: {role_stats}")
            
            # 格式化为带标签的文本
            formatted_text = self.parser.format_structured_text(structured)
            print(f"[Novel Text Structure] Formatting completed, output text length: {len(formatted_text)}")
            
            # 输出示例
            preview = formatted_text[:200] + "..." if len(formatted_text) > 200 else formatted_text
            print(f"[Novel Text Structure] Output text preview: {preview}")
            
            return (formatted_text,)
            
        except Exception as e:
            import traceback
            print(f"[Novel Text Structure] Processing failed: {e}")
            print(traceback.format_exc())
            # 失败时返回原文本
            return (novel_text,)
