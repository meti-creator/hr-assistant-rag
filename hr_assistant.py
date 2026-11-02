"""
Project 1: HR Assistant — Branch 1 (`step-1-policy-chunking`)
=============================================================
Goal for Step 1:
1. Create or load the company HR handbook (`hr_policy.txt`).
2. Split the handbook into small, overlapping word chunks so our
   search engine (in Step 2) can find exact policy rules without
   cutting sentences or losing context at the edges.
"""

from pathlib import Path
from typing import List

# Sample HR Handbook covering common employee questions & edge cases
DEFAULT_HR_POLICY = """ACME CORP EMPLOYEE HANDBOOK & HR POLICY (2026)

1. Annual Leave & Paid Time Off (PTO)
Full-time employees receive 20 days of paid annual leave per calendar year, accrued at 1.67 days per month. Part-time employees accrue annual leave on a pro-rata basis according to their contracted hours. Unused annual leave of up to 5 days may be carried over into the following calendar year, provided it is used by March 31. Leave requests of 3 or more consecutive days must be submitted in the HR portal at least 2 weeks in advance and approved by your direct manager.

2. Sick Leave & Medical Absence
Employees are entitled to up to 10 paid sick days per year for personal illness, medical appointments, or caring for an immediate family member. If you are absent for 3 or more consecutive working days due to illness, a medical certificate from a licensed healthcare provider is required. Unused sick leave does not carry over to the next year and is not paid out upon termination.

3. Working Hours & Remote Work Policy
Standard working hours are 9:00 AM to 5:00 PM, Monday through Friday, with a 1-hour unpaid lunch break. Acme Corp operates on a hybrid schedule: employees may work remotely up to 2 days per week (typically Tuesday and Thursday) after completing their initial 90-day probation period. Core collaboration hours for all staff—whether remote or in-office—are 10:00 AM to 3:00 PM local time. Fully remote arrangements require VP-level approval.

4. Public Holidays & Parental Leave
Acme Corp observes all statutory national public holidays. If a public holiday falls on a weekend, the following Monday is observed as a paid day off. Eligible employees who have completed at least 12 months of continuous service receive 16 weeks of paid primary caregiver parental leave or 4 weeks of paid secondary caregiver leave.

5. Expense Reimbursement & Professional Development
Work-related travel, client meals (up to $50 per person), and approved software subscriptions are reimbursable when submitted with itemized receipts within 30 days of purchase. Each full-time employee is also allocated an annual learning and development stipend of $1,000 for approved courses, books, or conference tickets.
"""


def load_or_create_policy(filepath: str = "hr_policy.txt") -> str:
    """
    Checks if `hr_policy.txt` exists in the project folder.
    If not, creates it using DEFAULT_HR_POLICY, then reads and returns the text.
    """
    path = Path(filepath)
    if not path.exists():
        path.write_text(DEFAULT_HR_POLICY.strip(), encoding="utf-8")
        print(f"Created new policy file at: {path.resolve()}")
    else:
        print(f"Loaded existing policy file from: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def split_into_chunks(
    text: str, max_words: int = 45, overlap_words: int = 10
) -> List[str]:
    """
    Splits the HR policy text into smaller pieces (chunks).
    
    Why overlap matters:
    If a rule sits right at the boundary of Chunk 2, repeating the last
    `overlap_words` (e.g., 10 words) at the start of Chunk 3 keeps the full
    thought intact.
    """
    # 1. Split by double newline so each policy section starts cleanly
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks: List[str] = []

    # 2. Slide a window across any section that is longer than `max_words`
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            chunks.append(section)
        else:
            step = max(1, max_words - overlap_words)
            for start in range(0, len(words), step):
                window = words[start : start + max_words]
                chunks.append(" ".join(window))
                if start + max_words >= len(words):
                    break

    return chunks


if __name__ == "__main__":
    # 1. Create/load the HR policy document
    policy_text = load_or_create_policy("hr_policy.txt")

    # 2. Split into chunks (45 words per chunk, 10 words overlap)
    chunks = split_into_chunks(policy_text, max_words=45, overlap_words=10)

    # 3. Inspect the output
    print(f"\nTotal chunks created: {len(chunks)}\n")
    for idx, chunk in enumerate(chunks, start=1):
        word_count = len(chunk.split())
        print(f"[Chunk {idx:02d}] ({word_count} words)")
        print(f"{chunk}\n")