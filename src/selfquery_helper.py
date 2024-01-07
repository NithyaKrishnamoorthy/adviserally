from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="features.brochure_url",
        description="URL of the insurance product brochure",
        type="string",
    ),
    AttributeInfo(
        name="features.currency_list",
        description="Available currencies for the insurance product",
        type="string",
    ),
    AttributeInfo(
        name="features.coverage_term_parameter",
        description="Number of years for insurance coverage",
        type="integer",
    ),
    AttributeInfo(
        name="features.coverage_term_type",
        description="Type of coverage term, e.g., 'fixed-term'",
        type="string",
    ),
    AttributeInfo(
        name="features.status",
        description="Current status of the insurance product, e.g., 'draft'",
        type="string",
    ),
    AttributeInfo(
        name="features.effective_date",
        description="Date when the insurance policy becomes effective",
        type="string",
    ),
    AttributeInfo(
        name="features.has_cash_value",
        description="Indicates if the policy has cash value",
        type="boolean",
    ),
    AttributeInfo(
        name="features.is_guaranteed_renewable",
        description="Indicates if the policy is guaranteed renewable",
        type="boolean",
    ),
    AttributeInfo(
        name="max coverage age",
        description="Maximum age for coverage",
        type="integer",
    ),
    AttributeInfo(
        name="features.has_death_benefit",
        description="Indicates if there is a death benefit",
        type="boolean",
    ),
    AttributeInfo(
        name="claim more than once on major illnesses",
        description="Indicates if multiple claims for major illnesses are allowed",
        type="boolean",
    ),
    AttributeInfo(
        name="features.multiple_pay_max_coverage_age",
        description="Maximum coverage age for multiple-pay scenarios",
        type="string",
    ),
    AttributeInfo(
        name="covers illnesses other than cancer",
        description="Indicates if illnesses besides cancer are covered",
        type="boolean",
    ),
    AttributeInfo(
        name="features.cancer_non_major_cap",
        description="Cap for non-major cancer conditions",
        type="string",
    ),
    AttributeInfo(
        name="features.cancer_overall_cap",
        description="Overall cap for cancer conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.heart_non_major_cap",
        description="Cap for non-major heart conditions",
        type="string",
    ),
    AttributeInfo(
        name="features.heart_overall_cap",
        description="Overall cap for heart conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.stroke_non_major_cap",
        description="Cap for non-major stroke conditions",
        type="string",
    ),
    AttributeInfo(
        name="features.stroke_overall_cap",
        description="Overall cap for stroke conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.other_non_major_cap",
        description="Cap for other non-major conditions",
        type="string",
    ),
    AttributeInfo(
        name="features.other_overall_cap",
        description="Overall cap for other conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.juvenile_overall_cap",
        description="Overall cap for juvenile conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.benefit_cap",
        description="Cap on benefits provided by the policy",
        type="float",
    ),
    AttributeInfo(
        name="features.has_non_guaranteed_component",
        description="Indicates if there is a non-guaranteed component in the policy",
        type="boolean",
    ),
    AttributeInfo(
        name="features.has_waiver_of_premium",
        description="Indicates if premium waiver is available",
        type="boolean",
    ),
    AttributeInfo(
        name="surrender benefit",
        description="Indicates if there is a surrender benefit",
        type="boolean",
    ),
    AttributeInfo(
        name="features.has_family_care_service",
        description="Indicates if family care services are included",
        type="boolean",
    ),
    AttributeInfo(
        name="features.has_second_medical_opinion_benefit",
        description="Indicates if a second medical opinion benefit is available",
        type="boolean",
    ),
    AttributeInfo(
        name="features.has_free_medical_checkup_benefit",
        description="Indicates if free medical checkups are included",
        type="boolean",
    ),
    AttributeInfo(
        name="features.number_of_major_critical_illness",
        description="Number of major critical illnesses covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.number_of_minor_critical_illness",
        description="Number of minor critical illnesses covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.official_number_of_major_critical_illness",
        description="Official number of major critical illnesses covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.official_number_of_minor_critical_illness",
        description="Official number of minor critical illnesses covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.official_number_of_juvenile_critical_illness",
        description="Official number of juvenile critical illnesses covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.cancer_major_per_claim",
        description="Coverage per claim for major cancer illnesses",
        type="float",
    ),
    AttributeInfo(
        name="features.cancer_minor_per_claim",
        description="Coverage per claim for minor cancer illnesses",
        type="float",
    ),
    AttributeInfo(
        name="features.heart_major_per_claim",
        description="Coverage per claim for major heart diseases",
        type="float",
    ),
    AttributeInfo(
        name="features.heart_minor_per_claim",
        description="Coverage per claim for minor heart diseases",
        type="float",
    ),
    AttributeInfo(
        name="features.stroke_major_per_claim",
        description="Coverage per claim for major stroke conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.stroke_minor_per_claim",
        description="Coverage per claim for minor stroke conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.other_major_per_claim",
        description="Coverage per claim for other major conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.other_minor_per_claim",
        description="Coverage per claim for other minor conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.juvenile_per_claim",
        description="Coverage per claim for juvenile conditions",
        type="float",
    ),
    AttributeInfo(
        name="features.number_severe",
        description="Number of severe conditions covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.number_intermediate",
        description="Number of intermediate conditions covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.number_early",
        description="Number of early-stage conditions covered",
        type="integer",
    ),
    AttributeInfo(
        name="features.number_additional_benefits",
        description="Number of additional benefits included",
        type="integer",
    ),
    AttributeInfo(
        name="features.number_special_benefits",
        description="Number of special benefits included",
        type="integer",
    ),
    AttributeInfo(
        name="name",
        description="Name of the insurance product",
        type="string",
    ),
    AttributeInfo(
        name="features.masked_name",
        description="Masked name of the insurance product",
        type="string",
    ),
    AttributeInfo(
        name="features.brochure_image",
        description="Image link for the insurance product brochure",
        type="string",
    ),
    AttributeInfo(
        name="features",
        description="Feature highlight of the insurance product",
        type="string",
    ),
    AttributeInfo(
        name="features.major_exclusion",
        description="Major exclusions of the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="features.product_risk",
        description="Risk level associated with the insurance product",
        type="string",
    ),
    AttributeInfo(
        name="features.underwriting_requirement",
        description="Underwriting requirements for the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="features.min_sum_assured",
        description="Minimum sum assured by the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="features.max_sum_assured",
        description="Maximum sum assured by the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="features.premium_adjust_frequency",
        description="Frequency of premium adjustments",
        type="string",
    ),
    AttributeInfo(
        name="features.premium_discount_info",
        description="Information about premium discounts",
        type="string",
    ),
    AttributeInfo(
        name="death benefit",
        description="Death benefit offered by the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="supplementary information for cancer benefits",
        description="Additional information on cancer benefits",
        type="string",
    ),
    AttributeInfo(
        name="supplementary information for heart disease",
        description="Additional information on heart disease benefits",
        type="string",
    ),
    AttributeInfo(
        name="supplementary information for stroke illnesses",
        description="Additional information on stroke benefits",
        type="string",
    ),
    AttributeInfo(
        name="supplementary information for other illnesses",
        description="Additional information on other illness benefits",
        type="string",
    ),
    AttributeInfo(
        name="supplmentary information for juvenile benefits",
        description="Additional information on juvenile benefits",
        type="string",
    ),
    AttributeInfo(
        name="features.investment_component",
        description="Investment component of the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="increase critical illness coverage without medical checkup",
        description="Option to increase critical illness coverage without medical checkup",
        type="string",
    ),
    AttributeInfo(
        name="supplementary benefit information",
        description="Information about supplementary benefits",
        type="string",
    ),
    AttributeInfo(
        name="rider information",
        description="Information about available riders with the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="features.currency_info",
        description="Information on currencies accepted for the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="how long you pay for",
        description="Duration for which premiums are payable",
        type="string",
    ),
    AttributeInfo(
        name="features.max_coverage_age_info",
        description="Information on maximum coverage age",
        type="string",
    ),
    AttributeInfo(
        name="how long you are covered for",
        description="Duration of coverage provided by the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="covered major illnesses",
        description="Details of major illnesses covered",
        type="string",
    ),
    AttributeInfo(
        name="covered non-major illnesses",
        description="Details of non-major illnesses covered",
        type="string",
    ),
    AttributeInfo(
        name="max possible claims (% of SA)",
        description="Maximum possible claims as a percentage of the Sum Assured",
        type="string",
    ),
    AttributeInfo(
        name="major cancer illness benefit (per claim)",
        description="Benefit for major cancer illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="non-major cancer illness benefit (per claim)",
        description="Benefit for non-major cancer illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="aggregate claims limit for all cancer illnesses",
        description="Aggregate claims limit for all cancer illnesses",
        type="string",
    ),
    AttributeInfo(
        name="major heart disease benefit (per claim)",
        description="Benefit for major heart diseases per claim",
        type="string",
    ),
    AttributeInfo(
        name="non-major heart disease benefit (per claim)",
        description="Benefit for non-major heart diseases per claim",
        type="string",
    ),
    AttributeInfo(
        name="aggregate claims limit for all heart diseases",
        description="Aggregate claims limit for all heart diseases",
        type="string",
    ),
    AttributeInfo(
        name="major stroke illness benefit (per claim)",
        description="Benefit for major stroke illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="non-major stroke illness benefit (per claim)",
        description="Benefit for non-major stroke illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="aggregate claims limit for all stroke illnesses",
        description="Aggregate claims limit for all stroke illnesses",
        type="string",
    ),
    AttributeInfo(
        name="major other illness benefit (per claim)",
        description="Benefit for other major illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="non-major other illness benefit (per claim)",
        description="Benefit for other non-major illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="aggregate claims limit for all other illnesses",
        description="Aggregate claims limit for all other illnesses",
        type="string",
    ),
    AttributeInfo(
        name="juvenile illness benefit (per claim)",
        description="Benefit for juvenile illnesses per claim",
        type="string",
    ),
    AttributeInfo(
        name="aggregate claims limit for all juvenile illnesses",
        description="Aggregate claims limit for all juvenile illnesses",
        type="string",
    ),
    AttributeInfo(
        name="insurer.icon",
        description="URL of the insurer's icon image",
        type="string",
    ),
    AttributeInfo(
        name="insurer.name",
        description="Name of the insurer",
        type="string",
    ),
    AttributeInfo(
        name="insurer.official_name",
        description="Official name of the insurer",
        type="string",
    ),
    AttributeInfo(
        name="insurer.fulfillment_ratio_website",
        description="Website URL showing the insurer's fulfillment ratio",
        type="string",
    ),
    AttributeInfo(
        name="insurer.official_website",
        description="Official website of the insurer",
        type="string",
    ),
    AttributeInfo(
        name="illness_definitions",
        description="Definitions of illnesses covered by the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="min_issue_age",
        description="minimum issue age for the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="max_issue_age",
        description="maximum issue age for the insurance policy",
        type="string",
    ),
    AttributeInfo(
        name="gender",
        description="Gender",
        type="boolean",
    ),
    AttributeInfo(
        name="smoker",
        description="Smoker",
        type="boolean",
    ),
    AttributeInfo(
        name="premium_term_parameter",
        description="Number of years for premium payment",
        type="boolean",
    ),
    AttributeInfo(
        name="premium_term_type",
        description="Type of premium payment term, e.g., 'fixed-term'",
        type="boolean",
    ),
    AttributeInfo(
        name="effective_date",
        description="Effective date of the insurance policy, when it's launched",
        type="boolean",
    ),
    AttributeInfo(
        name="status",
        description="status of the insurance policy, e.g., 'draft'",
        type="boolean",
    ),
    AttributeInfo(
        name="currency",
        description="Currency accepted for the insurance policy",
        type="boolean",
    ),
]
document_content_description = "Brief summary of a insurance product/policy"
