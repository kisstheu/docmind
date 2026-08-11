# DocMind Recruitment JD Constraints Plugin

该插件识别并提取单份、单岗位 JD 中明确写出的约束。程序调用方还可以在本次请求的
`DomainRequest.options` 中提供七字段显式求职规则；插件会把这些规则转换为内部不可变
`JobSearchRules`，复用现有三态比较（符合、冲突、未知）和固定 Markdown 渲染。

## 受控单 JD OCR 结构

显式使用 `职位名称：` 等标签的 canonical JD 保持原有识别与至少四字段合同。插件还支持
一类边界收紧的招聘页面 OCR 正文：首个非空行是视觉岗位标题，第二个非空行是唯一 K/千元
月薪区间，职责前的有限 header 中有唯一经验区间，随后各有且仅有一个职责 section 和一个
任职 section，且职责在任职之前。视觉标题中的文档自有拉丁 token 还必须在职责或任职正文
中复现；该证据来自当前文档，不依赖岗位、公司或技术栈词库。

OCR 路径只确定性提取视觉标题、独立薪资区间、header 中的经验区间和任职 section 内的
技术要求，并继续满足既有“至少四字段，且包含岗位名称与技术要求”门槛。地点与经验粘连、
学历或其他 metadata 的边界不确定时保持缺失，不做全局词语扫描或猜测。重复职责/任职块、
多 JD、简历式 section、损坏的 section heading、额外评价问句和证据不足的裸标题均 abstain。
页面尾部的联系人、公司卡片和地址区域不会进入技术要求。

## Structured options 契约

招聘插件只读取与稳定 `PLUGIN_ID` 相同的 namespace：
`org.docmind.recruitment.jd-constraints`。其他插件的 namespace 会被忽略，不改变原有 JD
提取结果。namespace 内业务 payload 的独立版本是 `OPTIONS_SCHEMA_VERSION = "1.0"`；
它不同于 Domain protocol 版本和插件 manifest 的 schema version。

规范 payload 如下，示例内容均为合成数据：

```python
{
    "org.docmind.recruitment.jd-constraints": {
        "schema_version": "1.0",
        "explicit_rules": {
            "minimum_monthly_salary_k": 13,
            "require_double_weekends": True,
            "allow_outsourcing": False,
            "allow_onsite": False,
            "allowed_locations": ["示例城市甲"],
            "candidate_education_level": "associate",
            "candidate_relevant_years": 2,
        },
    }
}
```

`explicit_rules` 只允许以下七个可选字段：

| 字段 | Wire 值 | 含义 |
| --- | --- | --- |
| `minimum_monthly_salary_k` | 有限、非负 number（不接受 bool） | 最低月薪，单位 K/月 |
| `require_double_weekends` | bool | 是否要求固定双休 |
| `allow_outsourcing` | bool | 是否接受外包、派遣或第三方签约 |
| `allow_onsite` | bool | 是否接受长期驻场 |
| `allowed_locations` | 非空字符串组成的 list | 可接受地点，按 OR 处理；空 list 合法 |
| `candidate_education_level` | string enum | 候选人X的学历层级 |
| `candidate_relevant_years` | 有限、非负 number（不接受 bool） | 候选人X的相关经验年限 |

学历枚举只接受 `unrestricted`、`secondary`、`associate`、`bachelor`、`master` 和
`doctorate`。数字使用其 JSON 十进制文本语义转换为 `Decimal`；地点保持原字符串、原顺序
和重复项。

namespace 对象必须且只能包含 `schema_version` 与 `explicit_rules`；未知字段、缺失字段、
错误类型、非法枚举、负数和非有限数字都会原子拒绝，不会部分执行或补默认值。

完全省略 options、传入空 options、只传其他 namespace，或传入合法的
`explicit_rules={}`，均保持原有 `## JD 明确约束` 结果逐字段不变。自身 namespace 非法时，
插件只在单 JD 已成功提取后返回固定 handled 提示：

```text
## 求职规则输入无效

本次未执行显式规则比较。请检查结构化求职规则后重试。
```

非 JD、跨域文本和多 JD 仍然 abstain，即使它们携带非法招聘 options。错误提示不会回显
字段名、规则值或完整 payload。

## Production StaticDomainHost 程序调用

```python
from app.domain_dispatch_port import dispatch_domain_request
from ask_notes import create_production_domain_host
from docmind_recruitment_plugin import PLUGIN_ID

jd_text = """请整理这份 JD 中明确写出的岗位约束
职位名称：合成服务开发工程师
岗位职责：负责某公司A的合成接口、测试流程、交付记录和技术文档维护。
任职要求：熟悉 Python；掌握 SQL；了解 FastAPI。
工作地点：示例城市甲
工作制：双休
经验要求：2 年以上
薪资：15K–20K"""

host = create_production_domain_host()
result = dispatch_domain_request(
    host,
    jd_text,
    options={
        PLUGIN_ID: {
            "schema_version": "1.0",
            "explicit_rules": {
                "minimum_monthly_salary_k": 13,
                "require_double_weekends": True,
                "allow_outsourcing": False,
            },
        },
    },
)
```

该入口经过真实 `dispatch_domain_request`、production `StaticDomainHost` 和
`RecruitmentJDPlugin.execute`。交互式 CLI 可通过通用
`--domain-options-file <path>` 读取完整的 `DomainRequest.options` JSON object；这不是招聘
专属参数，namespace、payload version `"1.0"` 与上述七字段契约保持不变。文件仅在进程
启动时读取一次，会话内不热加载，源文件由用户创建和维护。长期画像、简历自动读取、MCP、
多插件动态发现和长期配置仍未接入；本插件不会从环境变量、数据库、聊天历史或长期状态
隐式加载规则。
