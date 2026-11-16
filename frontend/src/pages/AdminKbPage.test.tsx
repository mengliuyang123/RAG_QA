import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExistingDocument, KbDocument, UploadOptions } from "../api/types";

const existing: ExistingDocument = {
  docId: "old", kbId: "ops", filename: "guide.md", status: "ready", uploadedAt: "2026-11-01T00:00:00Z",
};
const uploads: (UploadOptions | undefined)[] = [];
let documents: KbDocument[] = [];
let conflictOnFirst = true;
const batches: string[][] = [];
let failBatch: Error | null = null;

vi.mock("../state/AppContext", () => ({
  useApp: () => ({
    knowledgeBases: [{ id: "ops", name: "运维", description: "", icon: "books", documentsCount: 1 }],
    refreshKnowledgeBases: async () => undefined,
  }),
}));

vi.mock("../api", async () => {
  const { UploadConflictError } = await import("../api/mock");
  return {
    api: {
      supports: { insights: true, editKnowledgeBase: true },
      admin: {
        insights: async () => [],
        listDocuments: async () => documents,
        updateKnowledgeBase: async () => undefined,
        deleteDocument: async () => undefined,
        deleteDocuments: async (ids: string[]) => {
          batches.push(ids);
          if (failBatch) throw failBatch;
          documents = documents.filter((doc) => !ids.includes(doc.docId));
          return { deleted: ids.filter((id) => id !== "gone"), notFound: ids.filter((id) => id === "gone") };
        },
        uploadDocument: async (_kb: string, file: File, _progress: unknown, options?: UploadOptions) => {
          uploads.push(options);
          if (!options && conflictOnFirst) throw new UploadConflictError("知识库中已有同名文档", existing);
          if (!options) {
            return { kind: "exists", existing };
          }
          const document = {
            docId: "new", kbId: "ops", filename: file.name, fileType: "md", chunksCount: 1,
            updatedAt: null, fileStatus: "available" as const,
          };
          documents = [document];
          return { kind: "uploaded", document };
        },
      },
    },
  };
});

const { AdminKbPage } = await import("./AdminKbPage");
const { TooltipProvider } = await import("../components/ui/tooltip");

function renderPage() {
  render(<TooltipProvider><AdminKbPage /></TooltipProvider>);
}

afterEach(() => {
  cleanup();
  uploads.length = 0;
  documents = [];
  conflictOnFirst = true;
  batches.length = 0;
  failBatch = null;
  vi.restoreAllMocks();
});

function doc(docId: string, filename: string): KbDocument {
  return { docId, kbId: "ops", filename, fileType: "md", chunksCount: 1, updatedAt: null, fileStatus: "available" };
}

function choose(file: File) {
  fireEvent.change(screen.getByLabelText(/上传文档/), { target: { files: [file] } });
}

describe("AdminKbPage upload conflicts", () => {
  it("asks whether to replace on a name conflict and retries with the choice", async () => {
    renderPage();
    choose(new File(["v2"], "guide.md"));
    const dialog = await screen.findByRole("dialog");
    expect(dialog.textContent).toContain("guide.md");
    fireEvent.click(screen.getByRole("button", { name: "作为新版本替换" }));
    await waitFor(() => expect(screen.getByText("已用新版本替换 guide.md")).toBeTruthy());
    expect(uploads).toEqual([undefined, { onConflict: "replace", replaceDocId: "old" }]);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("keeps both documents when the admin saves it as another document", async () => {
    renderPage();
    choose(new File(["v2"], "guide.md"));
    fireEvent.click(await screen.findByRole("button", { name: "另存为一篇" }));
    await waitFor(() => expect(uploads).toEqual([undefined, { onConflict: "keep_both", replaceDocId: undefined }]));
  });

  it("cancels the upload without a second request", async () => {
    renderPage();
    choose(new File(["v2"], "guide.md"));
    fireEvent.click(await screen.findByRole("button", { name: "取消" }));
    await waitFor(() => expect(screen.getByText("已取消上传 guide.md")).toBeTruthy());
    expect(uploads).toEqual([undefined]);
  });

  it("reports an identical upload as already existing", async () => {
    conflictOnFirst = false;
    renderPage();
    choose(new File(["v1"], "copy.md"));
    await waitFor(() => expect(screen.getByText(/已存在/)).toBeTruthy());
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("AdminKbPage batch delete", () => {
  it("deletes the checked documents in one request after confirmation", async () => {
    documents = [doc("a", "a.md"), doc("b", "b.md"), doc("c", "c.md")];
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderPage();
    fireEvent.click(await screen.findByLabelText("选择 a.md"));
    fireEvent.click(screen.getByLabelText("选择 c.md"));
    expect(screen.getByText("已选 2 篇")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /删除所选/ }));
    await waitFor(() => expect(screen.getByText("已删除 2 篇文档")).toBeTruthy());
    expect(batches).toEqual([["a", "c"]]);
    expect(screen.queryByLabelText("选择 a.md")).toBeNull();
    expect(screen.getByLabelText("选择 b.md")).toBeTruthy();
    expect(screen.queryByText(/已选/)).toBeNull();
  });

  it("selects all, reports documents that were already gone, and splits batches at 100", async () => {
    documents = [...Array.from({ length: 101 }, (_, i) => doc(`d${i}`, `d${i}.md`)), doc("gone", "gone.md")];
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderPage();
    fireEvent.click(await screen.findByLabelText("全选文档"));
    expect(screen.getByText("已选 102 篇")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /删除所选/ }));
    await waitFor(() => expect(screen.getByText("已删除 101 篇文档，1 篇已不存在")).toBeTruthy());
    expect(batches.map((ids) => ids.length)).toEqual([100, 2]);
  });

  it("does nothing when the confirmation is declined and keeps the selection on failure", async () => {
    documents = [doc("a", "a.md"), doc("b", "b.md")];
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    renderPage();
    fireEvent.click(await screen.findByLabelText("选择 a.md"));
    fireEvent.click(screen.getByRole("button", { name: /删除所选/ }));
    expect(batches).toEqual([]);

    confirm.mockReturnValue(true);
    failBatch = new Error("文档仍在入库，请等待任务结束后删除");
    fireEvent.click(screen.getByRole("button", { name: /删除所选/ }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("仍在入库"));
    expect(screen.getByText("已选 1 篇")).toBeTruthy();
    expect(screen.getByLabelText("选择 a.md")).toBeTruthy();
  });
});
