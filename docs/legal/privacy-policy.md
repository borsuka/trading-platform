# Privacy Policy — PLACEHOLDER

> Not reviewed by a lawyer. Must be reviewed against GDPR, UK GDPR, CCPA and any other
> applicable regime before commercial use.

## What is collected

| Data | Why | Retention |
|---|---|---|
| Email address | Account identity, security notifications | Life of the account |
| Password | Authentication — stored only as an Argon2id hash, never in plaintext | Life of the account |
| Exchange API credentials | To place the orders you configure — stored encrypted (Fernet) | Until you remove the account |
| Trading history | To show your performance and support your own record-keeping | [Decide; 7 years is a common financial-records default] |
| Audit log | Security and accountability | [Decide] |
| IP address, user agent | Security, rate limiting, audit trail | [Decide] |
| Licence key, device fingerprint | Licence enforcement | Life of the licence |

## What is not collected

* Exchange **passwords** — the platform uses API keys only.
* Withdrawal-capable credentials — actively rejected at connection time.
* Payment card details — [handled by the payment processor; name it].

## Where it lives

In a self-hosted or desktop deployment, all trading data stays on **your** machine or server.
The vendor's licence server receives only what licence validation requires: the licence key, a
device fingerprint and the application version. It does not receive exchange credentials,
trading history, positions or balances.

[If a hosted deployment is offered, this section must be rewritten to describe it accurately.]

## Security measures

* Argon2id password hashing.
* Exchange credentials encrypted at rest, with the key stored separately from the ciphertext.
* Mandatory secret redaction in all logs.
* TLS in transit [operator must configure].
* Strict per-user data isolation, enforced in the data-access layer rather than by convention.

## Your rights

[GDPR/CCPA rights: access, rectification, erasure, portability, objection, restriction. Set out
how to exercise each and the response time.]

## Third parties

| Party | What they receive | Why |
|---|---|---|
| Your exchange | The orders you configure | To trade |
| News provider (optional) | The asset symbols you follow | To fetch news |
| [Payment processor] | [Billing details] | [Payment] |
| [Email provider] | [Email address] | [Transactional email] |

## Contact

[Data controller identity, address, and data-protection contact.]
