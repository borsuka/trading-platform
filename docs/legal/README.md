# Legal documents — PLACEHOLDERS

> **These are drafting aids, not legal documents.**
>
> They are written to show what a lawyer will need to address and to make the product's actual
> behaviour explicit. They have **not** been reviewed by a qualified lawyer, they are not
> tailored to any jurisdiction, and they must not be published as-is.
>
> Before any commercial launch, have all of them reviewed by a lawyer qualified in every
> jurisdiction you intend to sell into. Financial software is regulated differently almost
> everywhere, and the differences are not cosmetic.

## What is here

| File | Purpose |
|---|---|
| `terms-of-service.md` | The software licence and the limits of what it does |
| `privacy-policy.md` | What data is collected, why, and how long it is kept |
| `risk-disclosure.md` | What can go wrong, stated plainly |

## The distinctions a lawyer will care about

The architecture deliberately keeps these separate, and the documents should not blur them:

| What this is | What this is **not** |
|---|---|
| Software you run | A brokerage |
| Order routing to *your* exchange account | Custody of your funds |
| Rules you configure | Investment advice |
| A tool | Asset management |

The platform never takes custody. The customer holds their own funds in their own exchange
account and grants **trade-only** API access — the software rejects any key that can withdraw.
That structure is a deliberate design choice with legal consequences, and it should be
described accurately rather than minimised.

## Questions for counsel

1. Does distributing this software constitute regulated activity in the target jurisdictions?
2. Does providing strategies with configurable parameters constitute investment advice, or a
   personal recommendation, anywhere you intend to sell?
3. What disclosures are mandatory, and where must they appear?
4. What are the data-protection obligations (GDPR, UK GDPR, CCPA, and others) given that the
   platform stores email addresses and encrypted exchange credentials?
5. Are there restrictions on marketing automated trading tools to retail customers?
6. What limitation-of-liability language is actually enforceable in each jurisdiction?
7. Does displaying backtest results trigger performance-advertising rules?

## Language that must never appear

Not because of a style preference — because it is false, and in most jurisdictions saying it is
a regulatory problem:

* "Guaranteed profit" / "guaranteed monthly return"
* "Risk free" / "no risk"
* "AI knows where the market is going"
* "Cannot lose"
* Any past or projected return presented without the accompanying disclosure that past
  performance does not guarantee future performance
