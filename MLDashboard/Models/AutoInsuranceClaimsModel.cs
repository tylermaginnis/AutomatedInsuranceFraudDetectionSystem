public class AutoInsuranceClaimsModel
{
    public class Root
    {
        public string ClaimID { get; set; }
        public string PolicyID { get; set; }
        public EffectiveDates EffectiveDates { get; set; }
        public PolicyHolder PolicyHolder { get; set; }
        public AccidentDetails AccidentDetails { get; set; }
        public Coverage Coverage { get; set; }
        public string ClaimStatus { get; set; }
        public ClaimAmounts ClaimAmounts { get; set; }
        public AdjusterDetails AdjusterDetails { get; set; }
        public List<SupportingDocument> SupportingDocuments { get; set; }
        public List<ClaimHistory> ClaimHistory { get; set; }
        public PoliceReport PoliceReport { get; set; }
        public ClaimantFinancialInformation ClaimantFinancialInformation { get; set; }
        public ClaimantBehavior ClaimantBehavior { get; set; }
        public bool is_abnormal { get; set; }
        public string FraudLikelihood { get; set; }
        public List<double> FraudScore { get; set; }
    }

    public class EffectiveDates
    {
        public string StartDate { get; set; }
        public string EndDate { get; set; }
    }

    public class PolicyHolder
    {
        public string PolicyHolderID { get; set; }
        public string Name { get; set; }
        public string PolicyNumber { get; set; }
        public ContactInformation ContactInformation { get; set; }
    }

    public class ContactInformation
    {
        public string Phone { get; set; }
        public string Email { get; set; }
        public string Address { get; set; }
    }

    public class AccidentDetails
    {
        public string Date { get; set; }
        public string Location { get; set; }
        public string Description { get; set; }
    }

    public class Coverage
    {
        public BIL BIL { get; set; }
        public PDL PDL { get; set; }
        public PIP PIP { get; set; }
        public CollisionCoverage CollisionCoverage { get; set; }
        public ComprehensiveCoverage ComprehensiveCoverage { get; set; }
    }

    public class BIL
    {
        public int CoverageLimit { get; set; }
        public int ClaimedAmount { get; set; }
    }

    public class PDL
    {
        public int CoverageLimit { get; set; }
        public int ClaimedAmount { get; set; }
    }

    public class PIP
    {
        public int CoverageLimit { get; set; }
        public int ClaimedAmount { get; set; }
    }

    public class CollisionCoverage
    {
        public int CoverageLimit { get; set; }
        public int ClaimedAmount { get; set; }
    }

    public class ComprehensiveCoverage
    {
        public int CoverageLimit { get; set; }
        public int ClaimedAmount { get; set; }
    }

    public class ClaimAmounts
    {
        public int TotalClaimed { get; set; }
        public int TotalApproved { get; set; }
    }

    public class AdjusterDetails
    {
        public string Name { get; set; }
        public ContactInformation ContactInformation { get; set; }
    }

    public class SupportingDocument
    {
        public string DocumentType { get; set; }
        public string DocumentURL { get; set; }
    }

    public class ClaimHistory
    {
        public string Status { get; set; }
        public string Date { get; set; }
        public string Notes { get; set; }
    }

    public class PoliceReport
    {
        public string ReportID { get; set; }
        public string OfficerName { get; set; }
        public string ReportDetails { get; set; }
    }

    public class ClaimantFinancialInformation
    {
        public int CreditScore { get; set; }
        public double AnnualIncome { get; set; }
        public double DebtToIncomeRatio { get; set; }
    }

    public class ClaimantBehavior
    {
        public int ClaimFrequency { get; set; }
        public int LatePayments { get; set; }
        public int PolicyChanges { get; set; }
    }
}