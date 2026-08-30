"use client";

/**
 * Application shell: mode banner, navigation, auth gate.
 *
 * The mode banner is the most important element on the page. A user must never be uncertain
 * whether they are looking at paper or live money, so it sits above everything, updates from
 * every API response, and is visually distinct rather than merely labelled.
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import {
  clearTokens,
  currentMode,
  fetchInfo,
  isAuthenticated,
  logout,
  onModeChange,
  type TradingMode,
} from "@/lib/api";

const NAV: Array<{ section: string; items: Array<{ href: string; label: string }> }> = [
  {
    section: "Trading",
    items: [
      { href: "/", label: "Dashboard" },
      { href: "/bots", label: "Bots" },
      { href: "/positions", label: "Positions" },
      { href: "/orders", label: "Orders" },
      { href: "/trades", label: "Trades" },
    ],
  },
  {
    section: "Research",
    items: [
      { href: "/strategies", label: "Strategies" },
      { href: "/backtests", label: "Backtests" },
      { href: "/news", label: "News" },
    ],
  },
  {
    section: "Control",
    items: [
      { href: "/risk", label: "Risk" },
      { href: "/exchange-accounts", label: "Exchange accounts" },
      { href: "/license", label: "Licence" },
      { href: "/system", label: "System" },
    ],
  },
];

export function ModeBanner() {
  const [mode, setMode] = useState<TradingMode>(currentMode().mode);
  const [isLive, setIsLive] = useState(currentMode().isLive);

  useEffect(() => {
    fetchInfo()
      .then((info) => {
        setMode(info.mode);
        setIsLive(info.is_live);
      })
      .catch(() => undefined);
    return onModeChange((nextMode, nextIsLive) => {
      setMode(nextMode);
      setIsLive(nextIsLive);
    });
  }, []);

  return (
    <div className="mode-banner" data-live={isLive} role="status" aria-live="polite">
      <strong>{isLive ? "LIVE TRADING" : `${mode} MODE`}</strong>
      <span>
        {isLive
          ? "Real orders are being placed with real money."
          : "Simulated trading. No real money is at risk."}
      </span>
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!isAuthenticated()) {
      router.replace("/login");
      return;
    }
    setReady(true);
  }, [router]);

  async function handleSignOut() {
    await logout();
    clearTokens();
    router.replace("/login");
  }

  if (!ready) {
    return <div className="empty">Loading…</div>;
  }

  return (
    <>
      <ModeBanner />
      <div className="shell">
        <nav className="sidebar">
          <div className="sidebar__brand">Trading Platform</div>
          {NAV.map((group) => (
            <div key={group.section}>
              <div className="sidebar__section">{group.section}</div>
              {group.items.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="navlink"
                  data-active={
                    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
                  }
                >
                  {item.label}
                </Link>
              ))}
            </div>
          ))}
          <div style={{ marginTop: "auto", paddingTop: "1rem" }}>
            <button type="button" onClick={handleSignOut} style={{ width: "100%" }}>
              Sign out
            </button>
          </div>
        </nav>
        <main className="main">{children}</main>
      </div>
    </>
  );
}
