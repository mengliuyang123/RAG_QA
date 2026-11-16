import { useState } from "react";
import { api } from "../../api";
import type { DocumentReview } from "../../api/types";

export function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

/** 已过时，或复审日期已到。 */
export function needsAttention(review: DocumentReview | undefined): boolean {
  return !!review && (review.outdated || (!!review.reviewDueOn && review.reviewDueOn <= todayIso()));
}

export function ReviewChip({ review }: { review: DocumentReview | undefined }) {
  if (!review) return <span className="muted">—</span>;
  if (review.outdated) return <span className="chip chip-danger" title={review.note ?? "不再参与检索"}>已过时</span>;
  if (review.reviewDueOn && review.reviewDueOn <= todayIso()) {
    return <span className="chip chip-warning" title={review.note ?? undefined}>待复审</span>;
  }
  if (review.reviewDueOn) return <span className="chip" title={review.note ?? undefined}>{review.reviewDueOn} 复审</span>;
  return <span className="muted">—</span>;
}

// 设置复审日期、标记已过时。已过时的文档立即停止参与检索。
export function ReviewEditor({
  kbId, docId, filename, review, onSaved, onClose,
}: {
  kbId: string;
  docId: string;
  filename: string;
  review: DocumentReview | undefined;
  onSaved(review: DocumentReview): void;
  onClose(): void;
}) {
  const [dueOn, setDueOn] = useState(review?.reviewDueOn ?? "");
  const [outdated, setOutdated] = useState(review?.outdated ?? false);
  const [note, setNote] = useState(review?.note ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      onSaved(await api.admin.saveDocumentReview(kbId, docId, {
        reviewDueOn: dueOn || null, outdated, note: note.trim() || null,
      }));
      onClose();
    } catch (reason) {
      setError((reason as Error).message);
      setBusy(false);
    }
  };

  return (
    <div role="group" aria-label={`${filename} 的时效`} className="card" style={{ marginTop: 6, padding: 10 }}>
      <label className="field-label" htmlFor={`due-${docId}`} style={{ marginTop: 0 }}>复审日期</label>
      <input id={`due-${docId}`} type="date" className="text-input" value={dueOn} onChange={(event) => setDueOn(event.target.value)} />
      <label style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 8, fontSize: 13 }}>
        <input type="checkbox" checked={outdated} onChange={(event) => setOutdated(event.target.checked)} />
        已过时（不再参与检索）
      </label>
      <label className="field-label" htmlFor={`note-${docId}`}>说明</label>
      <input id={`note-${docId}`} className="text-input" maxLength={500} value={note} onChange={(event) => setNote(event.target.value)} />
      {error && <div role="alert" className="form-error">{error}</div>}
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" className="btn btn-primary" disabled={busy} onClick={save}>保存</button>
        <button type="button" className="btn" onClick={onClose}>取消</button>
      </div>
    </div>
  );
}
