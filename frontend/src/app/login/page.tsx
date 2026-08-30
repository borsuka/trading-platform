"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { ModeBanner } from "@/components/Shell";
import { Disclaimer, ErrorNotice, Notice } from "@/components/ui";
import { login, register } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setNotice(null);
    setBusy(true);
    try {
      if (mode === "register") {
        await register(email, password, fullName);
        await login(email, password);
      } else {
        await login(email, password);
      }
      router.replace("/");
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <ModeBanner />
      <div className="auth">
        <div className="auth__card card">
          <h1>{mode === "login" ? "Sign in" : "Create an account"}</h1>
          <ErrorNotice error={error} />
          {notice ? <Notice kind="info">{notice}</Notice> : null}

          <form onSubmit={submit}>
            {mode === "register" ? (
              <div className="field">
                <label htmlFor="name">Name</label>
                <input
                  id="name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  autoComplete="name"
                />
              </div>
            ) : null}

            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>

            <div className="field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                required
                minLength={12}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
              />
              {mode === "register" ? (
                <div className="field__hint">
                  At least 12 characters. A long passphrase beats a short complicated one.
                </div>
              ) : null}
            </div>

            <button type="submit" className="primary" disabled={busy} style={{ width: "100%" }}>
              {busy ? "Working…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>

          <p style={{ marginTop: "1rem", fontSize: "0.85rem" }}>
            {mode === "login" ? "No account? " : "Already registered? "}
            <button
              type="button"
              onClick={() => {
                setMode(mode === "login" ? "register" : "login");
                setError(null);
              }}
              style={{ border: "none", background: "none", padding: 0, color: "var(--accent)" }}
            >
              {mode === "login" ? "Create one" : "Sign in"}
            </button>
          </p>

          <Disclaimer />
        </div>
      </div>
    </>
  );
}
