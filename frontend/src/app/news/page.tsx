"use client";

import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Notice,
  PageHeader,
  useAsync,
} from "@/components/ui";
import { get } from "@/lib/api";

type NewsStatus = {
  enabled: boolean;
  provider: string;
  configured: boolean;
  message: string;
  policy: string;
};

type Article = {
  title: string;
  description: string | null;
  source: string;
  url: string | null;
  published_at: string;
  event_type: string;
  sentiment: string;
  impact: string;
  confidence: number;
  score: number;
};

export default function NewsPage() {
  return (
    <Shell>
      <News />
    </Shell>
  );
}

function News() {
  const [asset, setAsset] = useState("BTC");
  const status = useAsync(() => get<NewsStatus>("/api/v1/news/status"));
  const articles = useAsync(
    () => get<Article[]>(`/api/v1/news/articles?assets=${asset}&limit=50`).catch(() => []),
    [asset],
  );

  return (
    <>
      <PageHeader
        title="News"
        description="Classified news, deduplicated so one syndicated story counts once."
        actions={
          <button type="button" onClick={articles.reload}>
            Refresh
          </button>
        }
      />

      <ErrorNotice error={status.error} />

      {status.data ? (
        <Notice kind={status.data.configured ? "info" : "warn"}>
          <strong>{status.data.message}</strong>
          <div style={{ marginTop: "0.4rem" }}>{status.data.policy}</div>
        </Notice>
      ) : null}

      <div className="field" style={{ maxWidth: 200 }}>
        <label htmlFor="news-asset">Asset</label>
        <input
          id="news-asset"
          value={asset}
          onChange={(e) => setAsset(e.target.value.toUpperCase())}
        />
      </div>

      {articles.data && articles.data.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Published</th>
                <th>Headline</th>
                <th>Source</th>
                <th>Event</th>
                <th>Sentiment</th>
                <th>Impact</th>
                <th className="num">Score</th>
              </tr>
            </thead>
            <tbody>
              {articles.data.map((article, index) => (
                <tr key={index}>
                  <td className="mono muted">
                    {new Date(article.published_at).toLocaleString()}
                  </td>
                  <td style={{ whiteSpace: "normal", maxWidth: "40ch" }}>
                    {article.url ? (
                      <a href={article.url} target="_blank" rel="noopener noreferrer">
                        {article.title}
                      </a>
                    ) : (
                      article.title
                    )}
                  </td>
                  <td className="muted">{article.source}</td>
                  <td className="muted">{article.event_type.replace(/_/g, " ")}</td>
                  <td>
                    <span
                      className={`badge badge--${
                        article.sentiment.includes("positive")
                          ? "ok"
                          : article.sentiment.includes("negative")
                            ? "danger"
                            : "muted"
                      }`}
                    >
                      {article.sentiment.replace(/_/g, " ")}
                    </span>
                  </td>
                  <td>
                    <span
                      className={`badge badge--${
                        ["critical", "high"].includes(article.impact) ? "warn" : "muted"
                      }`}
                    >
                      {article.impact}
                    </span>
                  </td>
                  <td className="num">{article.score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>
          No news available. Running without a news provider is a fully supported configuration.
        </Empty>
      )}

      <Notice kind="info">
        The bundled classifier is rule-based, with hand-set keyword weights that have not been
        calibrated against realised price moves. Its confidence score means &ldquo;how clearly
        does this text match a known pattern&rdquo;, not &ldquo;how likely is this to move the
        market&rdquo;.
      </Notice>

      <Disclaimer />
    </>
  );
}
