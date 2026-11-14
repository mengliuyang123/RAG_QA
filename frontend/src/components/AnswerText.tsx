import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import type { ElementContent, Root } from "hast";
import type { Parent } from "unist";

import type { Citation } from "../api/types";
import { maskCitations } from "../lib/citations";

interface AnswerTextProps {
  answer: string;
  citations: Citation[];
  activePosition: number | null;
  onCite(citation: Citation): void;
}

// ⟦n⟧ 占位符在 markdown 解析后仍留在文本节点里；
// 这里把它拆成 <citationmark position="n"/> 元素，再由 components 映射渲染成引用按钮。
function splitMarkedText(value: string): ElementContent[] | null {
  const parts: ElementContent[] = [];
  let lastIndex = 0;
  for (const match of value.matchAll(/⟦(\d+)⟧/g)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      parts.push({ type: "text", value: value.slice(lastIndex, index) });
    }
    parts.push({
      type: "element",
      tagName: "citationmark",
      properties: { position: Number(match[1]) },
      children: [{ type: "text", value: match[1] }],
    });
    lastIndex = index + match[0].length;
  }
  if (parts.length === 0) return null;
  if (lastIndex < value.length) {
    parts.push({ type: "text", value: value.slice(lastIndex) });
  }
  return parts;
}

function rehypeCitationMarks() {
  return (tree: Root) => {
    const visit = (node: Parent) => {
      const children = node.children as ElementContent[];
      for (let index = children.length - 1; index >= 0; index -= 1) {
        const child = children[index];
        if (child.type === "element") {
          visit(child);
        } else if (child.type === "text") {
          const replacement = splitMarkedText(child.value);
          if (replacement) children.splice(index, 1, ...replacement);
        }
      }
    };
    visit(tree);
  };
}

// 正文中的 [n] 按 citation.position 匹配，而不是按数组下标；
// 找不到对应来源的编号显示为灰色，不可点击。
// 正文其余部分按 Markdown 渲染；react-markdown 默认不执行原始 HTML，防止注入。
export function AnswerText({ answer, citations, activePosition, onCite }: AnswerTextProps) {
  const byPosition = new Map(citations.map((citation) => [citation.position, citation]));
  // Components 的键类型只覆盖标准 HTML 标签，自定义 citationmark 需要断言；
  // 运行时 hast-util-to-jsx-runtime 会按 tagName 查 components 映射。
  const components = {
    citationmark: ({ position }: { position?: number | string }) => {
      const value = Number(position);
      const citation = byPosition.get(value);
      if (!citation) {
        return (
          <span className="cite-btn missing" title="没有找到这条引用的来源">
            {value}
          </span>
        );
      }
      return (
        <button
          type="button"
          className={`cite-btn ${activePosition === value ? "active" : ""}`}
          aria-label={`查看来源 ${value}：${citation.filename}`}
          onClick={() => onCite(citation)}
        >
          {value}
        </button>
      );
    },
  } as unknown as Components;
  return (
    <div className="answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        rehypePlugins={[rehypeCitationMarks]}
        components={components}
      >
        {maskCitations(answer)}
      </ReactMarkdown>
    </div>
  );
}

// 流式输出中的半成品回答：渲染 Markdown 排版，引用编号暂按普通文本显示。
export function PendingAnswer({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
      {text}
    </ReactMarkdown>
  );
}
