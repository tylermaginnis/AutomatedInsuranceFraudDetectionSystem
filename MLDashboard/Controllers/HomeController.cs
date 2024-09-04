using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using MLDashboard.Models;
using Newtonsoft.Json;
using System.IO;
using System.Linq;

namespace MLDashboard.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly List<AutoInsuranceClaimsModel.Root> _claims;

    public HomeController(ILogger<HomeController> logger)
    {
        _logger = logger;
        var jsonData = System.IO.File.ReadAllText("../Loader/Data/normalized.json");
        _claims = JsonConvert.DeserializeObject<List<AutoInsuranceClaimsModel.Root>>(jsonData);
    }

    public IActionResult Index()
    {
        var totalClaims = _claims.Count;

        // Claims by Coverage Type
        var claimsByCoverageType = _claims
            .Where(c => c.Coverage != null && c.ClaimAmounts?.TotalClaimed != null)
            .SelectMany(c => new List<(string CoverageType, int? ClaimedAmount)>
            {
                ("BIL", c.Coverage.BIL?.ClaimedAmount),
                ("PDL", c.Coverage.PDL?.ClaimedAmount),
                ("PIP", c.Coverage.PIP?.ClaimedAmount),
                ("CollisionCoverage", c.Coverage.CollisionCoverage?.ClaimedAmount),
                ("ComprehensiveCoverage", c.Coverage.ComprehensiveCoverage?.ClaimedAmount)
            })
            .Where(c => c.ClaimedAmount != null)
            .GroupBy(c => c.CoverageType)
            .Select(g => new { CoverageType = g.Key, TotalClaimedAmount = g.Sum(c => c.ClaimedAmount.Value) })
            .ToList();

        // Claims by Policy Holder
        var claimsByPolicyHolder = _claims
            .Where(c => c.PolicyHolder != null && c.ClaimAmounts?.TotalClaimed != null && c.ClaimAmounts?.TotalApproved != null)
            .GroupBy(c => c.PolicyHolder)
            .Select(g => new { 
                PolicyHolder = g.Key.Name, // Adjust this line if PolicyHolder is an object
                ClaimCount = g.Count(), 
                TotalClaimedAmount = g.Sum(c => c.ClaimAmounts.TotalClaimed),
                TotalApprovedAmount = g.Sum(c => c.ClaimAmounts.TotalApproved) 
            })
            .ToList();

        // Claims by Adjuster
        var claimsByAdjuster = _claims
            .Where(c => c.AdjusterDetails != null && c.ClaimAmounts?.TotalClaimed != null && c.ClaimAmounts?.TotalApproved != null)
            .GroupBy(c => c.AdjusterDetails.Name)
            .Select(g => new { 
                Adjuster = g.Key, 
                ClaimCount = g.Count(), 
                TotalClaimedAmount = g.Sum(c => c.ClaimAmounts.TotalClaimed),
                TotalApprovedAmount = g.Sum(c => c.ClaimAmounts.TotalApproved) 
            })
            .ToList();

        // Claims Over Time by Month
        var claimsOverTime = _claims
            .Where(c => c.AccidentDetails.Date != null)
            .GroupBy(c => new { Year = DateTime.Parse(c.AccidentDetails.Date).Year, Month = DateTime.Parse(c.AccidentDetails.Date).Month })
            .Select(g => new { 
                Date = new DateTime(g.Key.Year, g.Key.Month, 1), 
                ClaimCount = g.Count() 
            })
            .ToList();

        // Claims by Status Over Time
        var validStatuses = new[] { "Closed", "Filed", "In Review", "Approved" };

        var claimsByStatusOverTime = _claims
            .Where(c => c.ClaimStatus != null && c.ClaimHistory != null && validStatuses.Contains(c.ClaimStatus))
            .SelectMany(c => c.ClaimHistory, (c, h) => new { c.ClaimStatus, h.Date })
            .GroupBy(ch => new { ch.ClaimStatus, Year = DateTime.Parse(ch.Date).Year, Month = DateTime.Parse(ch.Date).Month })
            .Select(g => new { 
                g.Key.ClaimStatus, 
                Date = new DateTime(g.Key.Year, g.Key.Month, 1), 
                ClaimCount = g.Count() 
            })
            .ToList();

        // Average Claimed and Approved Amounts by Coverage Type
        var averageAmountsByCoverageType = _claims
            .Where(c => c.Coverage != null)
            .SelectMany(c => new List<(string CoverageType, int ClaimedAmount, int ApprovedAmount)>
            {
                ("BIL", c.Coverage.BIL?.ClaimedAmount ?? 0, c.ClaimAmounts?.TotalApproved ?? 0),
                ("PDL", c.Coverage.PDL?.ClaimedAmount ?? 0, c.ClaimAmounts?.TotalApproved ?? 0),
                ("PIP", c.Coverage.PIP?.ClaimedAmount ?? 0, c.ClaimAmounts?.TotalApproved ?? 0),
                ("CollisionCoverage", c.Coverage.CollisionCoverage?.ClaimedAmount ?? 0, c.ClaimAmounts?.TotalApproved ?? 0),
                ("ComprehensiveCoverage", c.Coverage.ComprehensiveCoverage?.ClaimedAmount ?? 0, c.ClaimAmounts?.TotalApproved ?? 0)
            })
            .GroupBy(c => c.CoverageType)
            .Select(g => new
            {
                Coverage = g.Key,
                AverageClaimedAmount = g.Average(c => c.ClaimedAmount),
                AverageApprovedAmount = g.Average(c => c.ApprovedAmount)
            })
            .ToList();

        // Claims by State
        var claimsByState = _claims
            .Where(c => c.AccidentDetails.Location != null)
            .GroupBy(c => c.AccidentDetails.Location.Split(',').Last().Trim().Split(' ').First()) // Split on ' ' and get the first part of the state string
            .Select(g => new { State = g.Key, ClaimCount = g.Count() })
            .ToList();
            
        // Claims by Policy Effective Dates
        var claimsByPolicyEffectiveDates = _claims
            .Where(c => c.EffectiveDates != null && c.EffectiveDates.StartDate != null)
            .GroupBy(c => DateTime.Parse(c.EffectiveDates.StartDate).Date)
            .Select(g => new { Date = g.Key, ClaimCount = g.Count() })
            .ToList();

        var model = new
        {
            TotalClaims = totalClaims,
            ClaimsByCoverageType = claimsByCoverageType,
            ClaimsByPolicyHolder = claimsByPolicyHolder,
            ClaimsByAdjuster = claimsByAdjuster,
            ClaimsOverTime = claimsOverTime,
            ClaimsByStatusOverTime = claimsByStatusOverTime,
            AverageAmountsByCoverageType = averageAmountsByCoverageType,
            ClaimsByState = claimsByState,
            ClaimsByPolicyEffectiveDates = claimsByPolicyEffectiveDates
        };

        ViewBag.ClaimsByCoverageTypeJson = JsonConvert.SerializeObject(claimsByCoverageType);
        ViewBag.ClaimsByPolicyHolderJson = JsonConvert.SerializeObject(claimsByPolicyHolder);
        ViewBag.ClaimsByAdjusterJson = JsonConvert.SerializeObject(claimsByAdjuster);
        ViewBag.ClaimsOverTimeJson = JsonConvert.SerializeObject(claimsOverTime);
        ViewBag.ClaimsByStatusOverTimeJson = JsonConvert.SerializeObject(claimsByStatusOverTime);
        ViewBag.AverageAmountsByCoverageTypeJson = JsonConvert.SerializeObject(averageAmountsByCoverageType);
        ViewBag.ClaimsByStateJson = JsonConvert.SerializeObject(claimsByState);
        ViewBag.ClaimsByPolicyEffectiveDatesJson = JsonConvert.SerializeObject(claimsByPolicyEffectiveDates);

        return View(model);
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
