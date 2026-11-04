import { Fragment } from "react";
import type { Citation } from "../api/types";
import { splitAnswer } from "../lib/citations";

interface AnswerTextProps {
  answer: string;
  citations: Citation[];
  activePosition: number | null;
  onCite(citation: Citation): void;
}

// 正文中的 [n] 按 citation.position 匹配，而不是按数组下标；
// 找不到对应来源的编号显示为灰色，不可点击。
export function AnswerText({ answer, citations, activePosition, onCite }: AnswerTextProps) {
  const byPosition = new Map(citations.map((citation) => [citation.position, citation]));
  return (
    <div className="answer">
      {splitAnswer(answer).map((segment, index) => {
        if (segment.kind === "text") return <Fragment key={index}>{segment.text}</Fragment>;
        const citation = byPosition.get(segment.position);
        if (!citation) {
          return (
            <span key={index} className="cite-btn missing" title="没有找到这条引用的来源">
              {segment.position}
            </span>
          );
        }
        return (
          <button
            key={index}
            type="button"
            className={`cite-btn ${activePosition === segment.position ? "active" : ""}`}
            aria-label={`查看来源 ${segment.position}：${citation.filename}`}
            onClick={() => onCite(citation)}
          >
            {segment.position}
          </button>
        );
      })}
    </div>
  );
}
