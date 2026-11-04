export type AnswerSegment =
  | { kind: "text"; text: string }
  | { kind: "citation"; position: number };

const CITATION_PATTERN = /\[(\d+)\]/g;

// 把回答切成文本与 [n] 引用片段，渲染时引用变成可点击按钮，
// 正文始终以纯文本输出，不经过 innerHTML。
export function splitAnswer(answer: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let lastIndex = 0;
  for (const match of answer.matchAll(CITATION_PATTERN)) {
    const index = match.index ?? 0;
    if (index > lastIndex) {
      segments.push({ kind: "text", text: answer.slice(lastIndex, index) });
    }
    segments.push({ kind: "citation", position: Number(match[1]) });
    lastIndex = index + match[0].length;
  }
  if (lastIndex < answer.length) {
    segments.push({ kind: "text", text: answer.slice(lastIndex) });
  }
  return segments;
}
