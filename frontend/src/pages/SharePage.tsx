import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import type { SharedTurn } from "../api/types";
import { AnswerText } from "../components/AnswerText";
import { Topbar } from "../components/Layout";
import { formatDate } from "./KnowledgePage";

// 打开同事分享的一轮回答。需要登录，链接有有效期。
export function SharePage() {
  const { token = "" } = useParams();
  const [shared, setShared] = useState<SharedTurn | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    api.getShare(token).then(setShared).catch((reason: Error) => setError(reason.message));
  }, [token]);

  return (
    <>
      <Topbar title="分享的回答" />
      <div className="page">
        <div className="page-inner">
          {error && <div className="banner" role="alert">{error}</div>}
          {!shared && !error && <div className="muted">正在加载…</div>}
          {shared && (
            <article className="card">
              <div className="secondary" style={{ fontSize: 12, marginBottom: 8 }}>
                {shared.sharedBy} 分享{shared.createdAt && ` · 回答于 ${formatDate(shared.createdAt)}`}
              </div>
              <h2 style={{ fontSize: 16, marginBottom: 10 }}>{shared.question}</h2>
              <div className="answer">
                <AnswerText
                  answer={shared.answer}
                  citations={shared.citations}
                  activePosition={active}
                  onCite={(citation) => setActive(citation.position)}
                />
              </div>
              {shared.citations.length > 0 && (
                <>
                  <div className="field-label">引用来源</div>
                  <ol style={{ margin: 0, paddingLeft: 18 }}>
                    {shared.citations.map((citation) => (
                      <li
                        key={citation.position}
                        style={{ fontWeight: active === citation.position ? 600 : undefined, marginBottom: 6 }}
                      >
                        {citation.filename}{citation.page !== null && ` · 第 ${citation.page} 页`}
                        {citation.chunkText && <div className="secondary" style={{ fontSize: 12 }}>{citation.chunkText}</div>}
                      </li>
                    ))}
                  </ol>
                </>
              )}
            </article>
          )}
        </div>
      </div>
    </>
  );
}
