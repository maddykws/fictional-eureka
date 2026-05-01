"""
Hard safety rules — evaluated BEFORE the LLM.

These are non-negotiable escalation/rejection rules that do not
depend on corpus retrieval.  The LLM cannot override them.

Rules:
  1. Prompt injection / jailbreak attempts
  2. Malicious requests (delete files, bypass security, etc.)
  3. Score / grade manipulation requests
  4. Refund / billing demands (simple "how to dispute" is OK)
  5. Identity theft / fraud
  6. Non-owner requesting account access restoration
  7. Infrastructure outage claims (site completely down)
  8. Assessment rescheduling (requires recruiter/company action)
  9. Requests to fill in company's internal security forms
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class SafetyDecision:
    triggered:    bool
    rule:         str        # which rule fired
    status:       str        # "escalated" or "replied"
    product_area: str
    response:     str
    justification: str
    request_type: str


# ── Pattern lists ──────────────────────────────────────────────────────────────

_PROMPT_INJECTION = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in [
        # Disclosure / reveal attempts
        r"(show|reveal|display|print|dump|output|tell me|give me).{0,40}(internal|system|prompt|rules|logic|instruction|document)",
        # Classic override patterns
        r"ignore (previous|above|prior|all) instruction",
        r"disregard (your|the) (previous|above|prior|all)",
        r"pretend (you are|to be|you're) (not|without)",
        r"jailbreak",
        r"dan mode",
        r"act as (if you have no|without any) (restriction|filter|rule|limit)",
        r"for\s+debugging\s+purposes?\s*,?\s*(show|reveal|display)",
        # HouYi context-partition injection (arXiv:2306.05499)
        r"new\s+task\s*:",
        r"system\s+prompt\s*:",
        r"you are now",
        r"your new (instruction|role|task|purpose|goal|directive)",
        r"---+\s*(new|override|system|admin|instruction)",
        r"###\s*(new|override|system|admin|instruction)",
        # rebuff heuristics (github.com/protectai/rebuff)
        r"\[system\]",
        r"<system>",
        r"<<SYS>>",
        r"respond only in",
        r"from now on (you (are|will)|ignore|forget)",
        # Multilingual patterns
        r"(affiche|montre|révèle).{0,40}(règles|documents|logique|interne)",  # French
        r"muéstrame.{0,40}(reglas|documentos|lógica|interna)",                 # Spanish
    ]
]

_MALICIOUS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bdelete\b.{0,30}\b(all|every|system|file)",
        r"\brm\s+-rf\b",
        r"\bformat\b.{0,30}\b(disk|drive|system)",
        r"\bbypass\b.{0,30}\b(security|auth|login|2fa)",
        r"\bhack\b",
        r"sql\s+inject",
        r"(execute|run).{0,20}(command|shell|script|code).{0,20}(system|admin|root)",
    ]
]

_SCORE_MANIPULATION = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(increase|change|update|fix|modify|adjust|raise|boost).{0,40}(score|grade|result|mark)",
        r"(tell|ask|force|make).{0,40}(company|recruiter|employer|hr).{0,40}(move|advance|proceed|hire)",
        r"graded.{0,30}unfairly",
        r"platform.{0,30}(wrong|incorrect|error).{0,20}(score|grade)",
        r"review my (answer|submission|test|code) and (fix|change|update|increase)",
    ]
]

_BILLING_REFUND = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(refund|give me my money|return (my )?money|money back)\b",
        r"\bpause\b.{0,20}\b(subscription|plan|billing|account)\b",
        r"\bcancel\b.{0,20}\b(subscription|plan)\b",
        r"\bpayment\b.{0,20}\b(order id|issue|problem|failed|stuck)\b",
        r"\border.{0,10}(id|#|number).{0,30}(cs_live|pay_|ch_)",
    ]
]

_FRAUD_IDENTITY = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bidentity.{0,20}(theft|stolen|compromised)\b",
        r"\bfraud\b",
        r"\bstolen.{0,30}(card|identity|account)\b",
        r"\bunauthori[sz]ed.{0,30}(transaction|charge|access)\b",
    ]
]

_NON_OWNER_ACCESS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(restore|give|grant|reinstate).{0,40}(access|account).{0,40}(not|no longer|even though).{0,30}(admin|owner)",
        r"(not|no longer|am not).{0,30}(admin|owner|workspace owner).{0,50}(restore|access|back)",
        r"restore.{0,30}access.{0,30}immediately",
    ]
]

_OUTAGE = [
    re.compile(p, re.IGNORECASE) for p in [
        r"site is down",
        r"(none|no|all) (of the )?pages? (are|is) (not )?accessible",
        r"(everything|all requests?|complete(ly)?).{0,20}(fail|broken|down|not working)",
        r"claude (has )?stopped working completely",
        r"stopped working.{0,10}completely",
    ]
]

_RESCHEDULE = [
    re.compile(p, re.IGNORECASE) for p in [
        r"reschedul(e|ing).{0,40}(assessment|test|interview)",
        r"alternative (date|time).{0,30}(assessment|test|interview)",
        r"(could not|unable|prevented).{0,30}(attend|take|complete).{0,30}(test|assessment|interview)",
    ]
]

_INFOSEC_FORMS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(fill(ing)?|complete|answer).{0,30}(infosec|security|compliance|vendor).{0,30}(form|questionnaire|assessment)",
        r"infosec process.{0,30}(company|form)",
    ]
]

_SUSPEND_BLOCK = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(card|account|service).{0,20}blocked",
        r"blocked.{0,20}(card|account)",
    ]
]


# ── Checker ───────────────────────────────────────────────────────────────────

def check(issue: str, subject: str, company: str) -> SafetyDecision | None:
    """
    Run all hard safety rules against the ticket text.
    Returns a SafetyDecision if any rule fires, else None.
    """
    text = f"{subject} {issue}".strip()
    co = company.strip().lower()

    # 1. Prompt injection
    for pat in _PROMPT_INJECTION:
        if pat.search(text):
            return SafetyDecision(
                triggered=True,
                rule="prompt_injection",
                status="escalated",
                product_area="security",
                response=(
                    "This request cannot be processed. It appears to contain instructions "
                    "that attempt to manipulate the support system. Please contact support "
                    "with a clear description of your actual issue."
                ),
                justification=(
                    "Detected prompt injection attempt — the ticket contains instructions "
                    "requesting disclosure of internal system logic or overriding agent rules. "
                    "Escalated for human review."
                ),
                request_type="invalid",
            )

    # 2. Malicious requests
    for pat in _MALICIOUS:
        if pat.search(text):
            return SafetyDecision(
                triggered=True,
                rule="malicious_request",
                status="escalated",
                product_area="security",
                response=(
                    "This request is outside the scope of our support services and cannot be fulfilled."
                ),
                justification=(
                    "Detected a potentially malicious or harmful request. "
                    "This falls outside support scope and has been flagged for security review."
                ),
                request_type="invalid",
            )

    # 3. Score manipulation
    for pat in _SCORE_MANIPULATION:
        if pat.search(text) and co in ("hackerrank", "none", ""):
            return SafetyDecision(
                triggered=True,
                rule="score_manipulation",
                status="escalated",
                product_area="screen",
                response=(
                    "We understand your concern, but HackerRank support is unable to modify "
                    "assessment scores or influence hiring decisions on behalf of candidates. "
                    "Assessment results are final and determined by the platform's automated grading. "
                    "If you believe there was a technical error, please contact the recruiter "
                    "or company directly."
                ),
                justification=(
                    "Request asks to change an assessment score or influence a recruiter decision — "
                    "both are outside support's authority and involve exam integrity. Escalated."
                ),
                request_type="invalid",
            )

    # 4. Billing / refunds (but NOT simple 'how do I dispute a charge' FAQs)
    _HOW_TO_DISPUTE = re.compile(
        r"how (do i|can i|to).{0,30}(dispute|report|raise|file).{0,20}(charge|transaction|claim)",
        re.IGNORECASE
    )
    for pat in _BILLING_REFUND:
        if pat.search(text) and not _HOW_TO_DISPUTE.search(text):
            domain_area = {
                "hackerrank": "billing",
                "claude":     "billing",
                "visa":       "general_support",
            }.get(co, "billing")
            return SafetyDecision(
                triggered=True,
                rule="billing_refund",
                status="escalated",
                product_area=domain_area,
                response=(
                    "Your billing or refund request requires review by our billing team. "
                    "Please contact support directly with your order details and account information "
                    "so they can assist you promptly."
                ),
                justification=(
                    "Ticket involves a billing, refund, or subscription change — these require "
                    "human review and account verification. Escalated to billing team."
                ),
                request_type="product_issue",
            )

    # 5. Fraud / identity theft
    # Exception: FAQ questions asking WHERE/HOW to report lost/stolen cards
    # are corpus-answerable ("where can I report...") — let them through.
    _HOW_TO_REPORT = re.compile(
        r"(where|how).{0,30}(report|report a|notify|block|cancel).{0,30}(lost|stolen|missing).{0,20}(card|visa)",
        re.IGNORECASE
    )
    for pat in _FRAUD_IDENTITY:
        if pat.search(text) and not _HOW_TO_REPORT.search(text):
            domain_area = "general_support" if co == "visa" else "security"
            return SafetyDecision(
                triggered=True,
                rule="fraud_identity",
                status="escalated",
                product_area=domain_area,
                response=(
                    "This is a serious situation that requires immediate human assistance. "
                    "Please contact your card issuer or our security team directly. "
                    "For Visa card fraud: call the Visa Global Customer Assistance Service at "
                    "+1 303 967 1090 (available 24/7) or the country-specific number on the back of your card."
                ),
                justification=(
                    "Ticket involves identity theft or fraud — high-risk situation requiring "
                    "immediate human escalation and cannot be resolved via automated support."
                ),
                request_type="product_issue",
            )

    # 6. Non-owner requesting account access restoration
    for pat in _NON_OWNER_ACCESS:
        if pat.search(text):
            return SafetyDecision(
                triggered=True,
                rule="non_owner_access",
                status="escalated",
                product_area="identity-management-sso-jit-scim",
                response=(
                    "Access restoration requests must be submitted by a workspace owner or admin. "
                    "Please ask your workspace admin to restore your access, or have your IT administrator "
                    "contact support with account verification details."
                ),
                justification=(
                    "Ticket requests account access restoration from a non-owner/non-admin user. "
                    "This requires admin verification and cannot be processed without account authority."
                ),
                request_type="product_issue",
            )

    # 7. Infrastructure outage
    for pat in _OUTAGE:
        if pat.search(text):
            return SafetyDecision(
                triggered=True,
                rule="infrastructure_outage",
                status="escalated",
                product_area="general-help",
                response=(
                    "We are sorry to hear you are experiencing issues accessing the service. "
                    "This appears to be a platform-level issue that requires investigation by our engineering team. "
                    "Please check the official status page for updates, and our team has been notified."
                ),
                justification=(
                    "Ticket reports complete service unavailability / outage — this is an infrastructure "
                    "issue that requires engineering escalation, not a support-answerable question."
                ),
                request_type="bug",
            )

    # 8. Assessment rescheduling
    for pat in _RESCHEDULE:
        if pat.search(text):
            return SafetyDecision(
                triggered=True,
                rule="reschedule_assessment",
                status="escalated",
                product_area="screen",
                response=(
                    "Assessment rescheduling cannot be arranged through HackerRank support directly. "
                    "Please reach out to the recruiter or hiring team at the company that invited you. "
                    "They have the ability to extend test deadlines or resend invitations."
                ),
                justification=(
                    "Assessment rescheduling requires action from the recruiting company — "
                    "HackerRank support cannot reschedule on behalf of candidates. Escalated."
                ),
                request_type="product_issue",
            )

    # 9. InfoSec form filling
    for pat in _INFOSEC_FORMS:
        if pat.search(text):
            return SafetyDecision(
                triggered=True,
                rule="infosec_forms",
                status="escalated",
                product_area="general-help",
                response=(
                    "Filling in your company's security or InfoSec questionnaires is handled by "
                    "HackerRank's enterprise security team, not support. Please contact your "
                    "HackerRank account manager or submit a request through the enterprise portal."
                ),
                justification=(
                    "Request is to complete an internal InfoSec or compliance form — "
                    "this requires enterprise security team involvement, not tier-1 support."
                ),
                request_type="feature_request",
            )

    return None  # no safety rule triggered
