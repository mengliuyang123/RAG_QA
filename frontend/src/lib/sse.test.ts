import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses complete events", () => {
    const parser = new SseParser();
    const messages = parser.push('event: status\ndata: {"phase":"deciding"}\n\nevent: token\ndata: {"token":"你好"}\n\n');
    expect(messages).toEqual([
      { event: "status", data: '{"phase":"deciding"}' },
      { event: "token", data: '{"token":"你好"}' },
    ]);
  });

  it("buffers events split across network chunks", () => {
    const parser = new SseParser();
    expect(parser.push("event: tok")).toEqual([]);
    expect(parser.push('en\ndata: {"tok')).toEqual([]);
    expect(parser.push('en":"a"}\n')).toEqual([]);
    expect(parser.push("\n")).toEqual([{ event: "token", data: '{"token":"a"}' }]);
  });

  it("handles CRLF, comments, and multi-line data", () => {
    const parser = new SseParser();
    const messages = parser.push(": keep-alive\r\n\r\nevent: complete\r\ndata: line1\r\ndata: line2\r\n\r\n");
    expect(messages).toEqual([{ event: "complete", data: "line1\nline2" }]);
  });

  it("defaults the event name and flushes a trailing event", () => {
    const parser = new SseParser();
    expect(parser.push("data: x")).toEqual([]);
    expect(parser.flush()).toEqual([{ event: "message", data: "x" }]);
  });
});
