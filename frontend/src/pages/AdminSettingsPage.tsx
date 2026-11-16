import { CheckCircle2Icon, EyeIcon, EyeOffIcon, InfoIcon, RotateCcwIcon, TriangleAlertIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { ModelSettingField, ModelSettings } from "../api/types";
import { Topbar } from "../components/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { TooltipIconButton } from "@/components/ui/tooltip-icon-button";
import { cn } from "@/lib/utils";

type Draft = Record<string, string>;

const EMBEDDING_FIELDS = new Set([
  "embedding_api_key", "embedding_base_url", "embedding_model", "embedding_dimensions", "openai_api_key",
]);

const asText = (value: string | number | null) => (value === null ? "" : String(value));

function toDraft(settings: ModelSettings): Draft {
  const draft: Draft = {};
  for (const group of settings.groups) {
    for (const field of group.fields) draft[field.name] = asText(field.value);
  }
  return draft;
}

function SettingInput({
  field, value, onChange,
}: { field: ModelSettingField; value: string; onChange(value: string): void }) {
  const [revealed, setRevealed] = useState(false);
  const secret = field.kind === "secret";
  const placeholder = field.defaultValue !== null && field.defaultValue !== ""
    ? `默认：${field.defaultValue}`
    : secret ? "未设置" : "未设置";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor={`setting-${field.name}`} className="text-sm font-medium">{field.label}</label>
        <code className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">{field.envKey}</code>
        {field.overriddenByEnvironment && (
          <span
            className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[11px] text-amber-700 dark:text-amber-300"
            title="进程环境变量里也设置了这一项。现在保存会立即生效，但后端重启后会以环境变量为准。"
          >
            重启后以环境变量为准
          </span>
        )}
      </div>
      <div className="relative">
        <Input
          id={`setting-${field.name}`}
          type={secret && !revealed ? "password" : field.kind === "int" ? "number" : "text"}
          inputMode={field.kind === "int" ? "numeric" : undefined}
          min={field.kind === "int" ? 1 : undefined}
          autoComplete="off"
          spellCheck={false}
          value={value}
          placeholder={placeholder}
          className={cn("font-mono text-sm", secret && "pe-10")}
          onChange={(event) => onChange(event.target.value)}
        />
        {secret && (
          <TooltipIconButton
            tooltip={revealed ? "隐藏" : "显示明文"}
            aria-label={revealed ? `隐藏 ${field.label}` : `显示 ${field.label}`}
            className="absolute end-1.5 top-1/2 size-7 -translate-y-1/2"
            onClick={() => setRevealed(!revealed)}
          >
            {revealed ? <EyeOffIcon /> : <EyeIcon />}
          </TooltipIconButton>
        )}
      </div>
      {field.hint && <p className="m-0 text-xs text-muted-foreground">{field.hint}</p>}
    </div>
  );
}

export function AdminSettingsPage() {
  const [settings, setSettings] = useState<ModelSettings | null>(null);
  const [draft, setDraft] = useState<Draft>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<Date | null>(null);

  useEffect(() => {
    api.admin.getModelSettings()
      .then((result) => {
        setSettings(result);
        setDraft(toDraft(result));
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);

  const original = useMemo(() => (settings ? toDraft(settings) : {}), [settings]);
  const changed = Object.keys(draft).filter((name) => draft[name].trim() !== original[name]);

  const save = async () => {
    if (!settings || changed.length === 0) return;
    if (
      changed.some((name) => EMBEDDING_FIELDS.has(name))
      && !window.confirm("将修改向量模型配置。保存前会请求一次向量接口做连接测试；如果新维度与已入库的向量不一致，保存会被拒绝。继续吗？")
    ) return;
    setSaving(true);
    setSaveError(null);
    try {
      const values: Record<string, string | null> = {};
      for (const name of changed) values[name] = draft[name].trim() === "" ? null : draft[name].trim();
      const result = await api.admin.updateModelSettings(values);
      setSettings(result);
      setDraft(toDraft(result));
      setSavedAt(new Date());
    } catch (error) {
      setSaveError((error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <Topbar title="模型配置" />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 pt-2 pb-28">
          <div className="flex items-start gap-2 rounded-lg border bg-muted/40 p-3 text-sm text-muted-foreground">
            <InfoIcon className="mt-0.5 size-4 shrink-0" />
            <div>
              保存后立即生效：之后的新提问使用新配置，正在进行的回答不受影响。配置同时写入服务器上的
              <code className="mx-1 rounded bg-muted px-1 text-xs">.env</code>
              文件，重启后仍然有效。
              {settings && (
                <span className="mt-1 block">
                  当前向量索引：
                  {settings.storedEmbeddingDimensions === null
                    ? "还没有入库任何向量，可以自由更换向量模型和维度。"
                    : `${settings.storedEmbeddingDimensions} 维。更换向量模型或维度时，新维度必须与之一致。`}
                </span>
              )}
            </div>
          </div>

          {loadError && (
            <div role="alert" className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">{loadError}</div>
          )}
          {!settings && !loadError && (
            <div className="flex flex-col gap-4">
              {[0, 1, 2].map((index) => <Skeleton key={index} className="h-40 w-full rounded-xl" />)}
            </div>
          )}

          {settings?.groups.map((group) => (
            <section key={group.id} aria-label={group.title} className="rounded-xl border p-4">
              <h2 className="m-0 text-base font-medium">{group.title}</h2>
              <p className="m-0 mt-1 text-sm text-muted-foreground">{group.description}</p>
              <div className="mt-4 flex flex-col gap-4">
                {group.fields.map((field) => (
                  <SettingInput
                    key={field.name}
                    field={field}
                    value={draft[field.name] ?? ""}
                    onChange={(value) => {
                      setDraft((current) => ({ ...current, [field.name]: value }));
                      setSaveError(null);
                      setSavedAt(null);
                    }}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>

      {settings && (changed.length > 0 || saveError || savedAt) && (
        <div className="pointer-events-none fixed inset-x-0 bottom-4 z-20 flex justify-center px-4 md:left-64">
          <div className="pointer-events-auto flex w-full max-w-3xl flex-wrap items-center gap-3 rounded-xl border bg-background p-3 shadow-lg duration-200 animate-in fade-in slide-in-from-bottom-2">
            {saveError ? (
              <span role="alert" className="flex min-w-0 flex-1 items-start gap-2 text-sm text-destructive">
                <TriangleAlertIcon className="mt-0.5 size-4 shrink-0" />
                {saveError}
              </span>
            ) : changed.length > 0 ? (
              <span className="flex-1 text-sm text-muted-foreground">{changed.length} 项修改未保存</span>
            ) : (
              <span role="status" className="flex flex-1 items-center gap-2 text-sm text-emerald-700 dark:text-emerald-300">
                <CheckCircle2Icon className="size-4" />
                已保存，新配置已生效
              </span>
            )}
            {changed.length > 0 && (
              <>
                <Button variant="ghost" size="sm" disabled={saving} onClick={() => { setDraft(original); setSaveError(null); }}>
                  <RotateCcwIcon />
                  撤销
                </Button>
                <Button size="sm" disabled={saving} onClick={() => void save()}>
                  {saving ? "正在保存…" : "保存并生效"}
                </Button>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
