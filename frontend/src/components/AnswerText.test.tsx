import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Citation } from "../api/types";
import { splitAnswer } from "../lib/citations";
import { AnswerText } from "./AnswerText";

afterEach(cleanup);

function makeCitation(position: number, filename: string): Citation {
  return {
    position,
    kbId: null,
    docId: `doc-${position}`,
    chunkId: null,
    filename,
    page: null,
    chunkText: "片段",
    docUpdatedAt: null,
  };
}

describe("splitAnswer", () => {
  it("splits text and citation markers", () => {
    expect(splitAnswer("A [2] B [10]")).toEqual([
      { kind: "text", text: "A " },
      { kind: "citation", position: 2 },
      { kind: "text", text: " B " },
      { kind: "citation", position: 10 },
    ]);
  });
});

describe("AnswerText", () => {
  it("matches markers by position, not list index", () => {
    const onCite = vi.fn();
    const second = makeCitation(5, "b.md");
    render(
      <AnswerText
        answer="先看 [5]，再看 [2]。"
        citations={[makeCitation(2, "a.md"), second]}
        activePosition={null}
        onCite={onCite}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "查看来源 5：b.md" }));
    expect(onCite).toHaveBeenCalledWith(second);
  });

  it("renders markers without a source as non-interactive", () => {
    render(<AnswerText answer="没有来源 [3]" citations={[]} activePosition={null} onCite={() => undefined} />);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByTitle("没有找到这条引用的来源").textContent).toBe("3");
  });

  it("never interprets answer text as HTML", () => {
    const { container } = render(
      <AnswerText answer={'<img src=x onerror="alert(1)">'} citations={[]} activePosition={null} onCite={() => undefined} />,
    );
    expect(container.querySelector("img")).toBeNull();
  });
});
