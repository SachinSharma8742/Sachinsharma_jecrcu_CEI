// Embedded Model Parameters (trained on X Education Lead Scoring dataset)
const ModelParams = {
    scaler: {
        features: ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"],
        mean: [3.331014223871367, 483.8758503401361, 2.3187755102040817],
        scale: [3.257413848819591, 546.5234096048471, 1.9642748957608736]
    },
    model_no_tags: {
        intercept: -0.20608281220043953,
        coefficients: {
            "Do Not Email": -1.095124457155983,
            "Total Time Spent on Website": 1.0840745626582775,
            "Lead Origin_Landing Page Submission": -1.1154242672039916,
            "Lead Origin_Lead Add Form": 3.34398334815693,
            "Lead Source_Olark Chat": 0.973769876250683,
            "Lead Source_Welingak Website": 2.4256609194654244,
            "Last Activity_Converted to Lead": -1.0125374292246132,
            "Last Activity_Email Bounced": -1.3497664081581082,
            "Last Activity_Olark Chat Conversation": -1.2354529390863147,
            "Last Activity_Others": -1.4387354286098357,
            "Specialization_Hospitality Management": -1.024634674925783,
            "Specialization_Not Specified": -1.1229387934031219,
            "What is your current occupation_Working Professional": 2.636151876557912,
            "Last Notable Activity_Email Bounced": 1.620656676832641,
            "Last Notable Activity_Had a Phone Conversation": 3.3169086423519047,
            "Last Notable Activity_Modified": -0.2860287208031526,
            "Last Notable Activity_SMS Sent": 1.3998157035814403,
            "Last Notable Activity_Unreachable": 1.5314103268199921
        },
        optimal_cutoff: 0.4
    },
    model_with_tags: {
        intercept: -2.7949783824207244,
        coefficients: {
            "Total Time Spent on Website": 0.931209769501426,
            "Lead Origin_Landing Page Submission": -1.0799091293660106,
            "Lead Source_Welingak Website": 4.676883332752102,
            "Last Activity_SMS Sent": 2.038553976070193,
            "Tags_Busy": 2.5797112916881817,
            "Tags_Closed by Horizzon": 9.855690184665459,
            "Tags_Lost to EINS": 8.476263829812616,
            "Tags_Not Specified": 2.024788825533186,
            "Tags_Ringing": -1.2943954237245807,
            "Tags_Want to take admission but has financial problems": 3.3280295289827198,
            "Tags_Will revert after reading the email": 6.733274154842595,
            "Tags_in touch with EINS": 2.7080080468017784,
            "Tags_switched off": -1.5820215088330718,
            "Last Notable Activity_Email Link Clicked": -1.2354883502235554,
            "Last Notable Activity_Modified": -1.7620432113477256,
            "Last Notable Activity_Olark Chat Conversation": -1.4515548802137581
        },
        optimal_cutoff: 0.3
    }
};

// Global App State
let leadsData = [];
let filteredLeads = [];
let currentPage = 1;
const pageSize = 10;
let activeTab = 'no_tags'; // 'no_tags' or 'with_tags'
let scoreChart = null;
let sourceChart = null;

// Initial Mock Data (to show before file upload)
const mockLeads = [
    { "Lead Number": 660727, "Lead Origin": "Landing Page Submission", "Lead Source": "Direct Traffic", "Do Not Email": "No", "TotalVisits": 2, "Total Time Spent on Website": 1532, "Page Views Per Visit": 2.0, "Last Activity": "Email Opened", "Specialization": "Business Administration", "What is your current occupation": "Student", "Tags": "Will revert after reading the email", "Last Notable Activity": "Email Opened" },
    { "Lead Number": 660681, "Lead Origin": "Landing Page Submission", "Lead Source": "Google", "Do Not Email": "No", "TotalVisits": 2, "Total Time Spent on Website": 1428, "Page Views Per Visit": 1.0, "Last Activity": "Converted to Lead", "Specialization": "Select", "What is your current occupation": "Unemployed", "Tags": "Will revert after reading the email", "Last Notable Activity": "Modified" },
    { "Lead Number": 660562, "Lead Origin": "API", "Lead Source": "Organic Search", "Do Not Email": "No", "TotalVisits": 11, "Total Time Spent on Website": 1538, "Page Views Per Visit": 11.0, "Last Activity": "Email Opened", "Specialization": "Marketing Management", "What is your current occupation": "Unemployed", "Tags": "Will revert after reading the email", "Last Notable Activity": "Modified" },
    { "Lead Number": 660737, "Lead Origin": "API", "Lead Source": "Olark Chat", "Do Not Email": "No", "TotalVisits": 0, "Total Time Spent on Website": 0, "Page Views Per Visit": 0.0, "Last Activity": "Page Visited on Website", "Specialization": "Select", "What is your current occupation": "Unemployed", "Tags": "Interested in other courses", "Last Notable Activity": "Modified" },
    { "Lead Number": 660558, "Lead Origin": "Landing Page Submission", "Lead Source": "Organic Search", "Do Not Email": "No", "TotalVisits": 5, "Total Time Spent on Website": 170, "Page Views Per Visit": 5.0, "Last Activity": "Email Opened", "Specialization": "Business Administration", "What is your current occupation": "Unemployed", "Tags": "Ringing", "Last Notable Activity": "Email Opened" },
    { "Lead Number": 660471, "Lead Origin": "Landing Page Submission", "Lead Source": "Google", "Do Not Email": "No", "TotalVisits": 1, "Total Time Spent on Website": 1013, "Page Views Per Visit": 1.0, "Last Activity": "Converted to Lead", "Specialization": "Banking, Investment And Insurance", "What is your current occupation": "Unemployed", "Tags": "Will revert after reading the email", "Last Notable Activity": "Modified" },
    { "Lead Number": 660458, "Lead Origin": "API", "Lead Source": "Google", "Do Not Email": "No", "TotalVisits": 6, "Total Time Spent on Website": 1137, "Page Views Per Visit": 1.5, "Last Activity": "Email Opened", "Specialization": "Marketing Management", "What is your current occupation": "Unemployed", "Tags": "Will revert after reading the email", "Last Notable Activity": "Email Opened" },
    { "Lead Number": 660312, "Lead Origin": "Landing Page Submission", "Lead Source": "Google", "Do Not Email": "No", "TotalVisits": 4, "Total Time Spent on Website": 1622, "Page Views Per Visit": 4.0, "Last Activity": "Unreachable", "Specialization": "Operations Management", "What is your current occupation": "Unemployed", "Tags": "Ringing", "Last Notable Activity": "Modified" },
    { "Lead Number": 660267, "Lead Origin": "Landing Page Submission", "Lead Source": "Google", "Do Not Email": "No", "TotalVisits": 5, "Total Time Spent on Website": 563, "Page Views Per Visit": 5.0, "Last Activity": "Email Opened", "Specialization": "Finance Management", "What is your current occupation": "Working Professional", "Tags": "Will revert after reading the email", "Last Notable Activity": "Email Opened" },
    { "Lead Number": 660149, "Lead Origin": "API", "Lead Source": "Google", "Do Not Email": "No", "TotalVisits": 6, "Total Time Spent on Website": 1225, "Page Views Per Visit": 6.0, "Last Activity": "Page Visited on Website", "Specialization": "Supply Chain Management", "What is your current occupation": "Unemployed", "Tags": "Diploma holder (Not Eligible)", "Last Notable Activity": "Modified" }
];

// Single lead evaluation function
function scoreLead(lead) {
    // 1. Imputation & default values
    const specialization = (lead["Specialization"] || "Not Specified") === "Select" ? "Not Specified" : (lead["Specialization"] || "Not Specified");
    const occupation = lead["What is your current occupation"] || "Unemployed";
    const leadSource = lead["Lead Source"] || "Google";
    const lastActivity = lead["Last Activity"] || "Others";
    const tags = (lead["Tags"] || "Not Specified") === "Select" ? "Not Specified" : (lead["Tags"] || "Not Specified");
    const leadOrigin = lead["Lead Origin"] || "API";
    const lastNotableAct = lead["Last Notable Activity"] || "Modified";
    
    const doNotEmail = (lead["Do Not Email"] || "No") === "Yes" ? 1 : 0;
    
    // Cap numerical features at training thresholds (TotalVisits=17, Page Views=9.0)
    let totalVisits = parseFloat(lead["TotalVisits"] || 0);
    if (totalVisits > 17) totalVisits = 17;
    
    let timeSpent = parseFloat(lead["Total Time Spent on Website"] || 0);
    
    let pageViews = parseFloat(lead["Page Views Per Visit"] || 0);
    if (pageViews > 9) pageViews = 9.0;
    
    // Scale numerical columns using saved parameters
    const s_mean = ModelParams.scaler.mean;
    const s_scale = ModelParams.scaler.scale;
    
    const scaled_visits = (totalVisits - s_mean[0]) / s_scale[0];
    const scaled_time = (timeSpent - s_mean[1]) / s_scale[1];
    const scaled_views = (pageViews - s_mean[2]) / s_scale[2];

    // Dummy helper
    const checkDummy = (colName, value, activeVal) => (value === activeVal) ? 1 : 0;

    // 2. Score Model A (No Tags)
    let z_a = ModelParams.model_no_tags.intercept;
    const coef_a = ModelParams.model_no_tags.coefficients;
    
    z_a += doNotEmail * (coef_a["Do Not Email"] || 0);
    z_a += scaled_time * (coef_a["Total Time Spent on Website"] || 0);
    z_a += checkDummy("Lead Origin", leadOrigin, "Landing Page Submission") * (coef_a["Lead Origin_Landing Page Submission"] || 0);
    z_a += checkDummy("Lead Origin", leadOrigin, "Lead Add Form") * (coef_a["Lead Origin_Lead Add Form"] || 0);
    z_a += checkDummy("Lead Source", leadSource, "Olark Chat") * (coef_a["Lead Source_Olark Chat"] || 0);
    z_a += checkDummy("Lead Source", leadSource, "Welingak Website") * (coef_a["Lead Source_Welingak Website"] || 0);
    z_a += checkDummy("Last Activity", lastActivity, "Converted to Lead") * (coef_a["Last Activity_Converted to Lead"] || 0);
    z_a += checkDummy("Last Activity", lastActivity, "Email Bounced") * (coef_a["Last Activity_Email Bounced"] || 0);
    z_a += checkDummy("Last Activity", lastActivity, "Olark Chat Conversation") * (coef_a["Last Activity_Olark Chat Conversation"] || 0);
    z_a += (["Converted to Lead", "Email Bounced", "Olark Chat Conversation", "Email Opened", "SMS Sent", "Page Visited on Website", "Unreachable"].includes(lastActivity) ? 0 : 1) * (coef_a["Last Activity_Others"] || 0);
    z_a += checkDummy("Specialization", specialization, "Hospitality Management") * (coef_a["Specialization_Hospitality Management"] || 0);
    z_a += checkDummy("Specialization", specialization, "Not Specified") * (coef_a["Specialization_Not Specified"] || 0);
    z_a += checkDummy("Occupation", occupation, "Working Professional") * (coef_a["What is your current occupation_Working Professional"] || 0);
    z_a += checkDummy("Last Notable Activity", lastNotableAct, "Email Bounced") * (coef_a["Last Notable Activity_Email Bounced"] || 0);
    z_a += checkDummy("Last Notable Activity", lastNotableAct, "Had a Phone Conversation") * (coef_a["Last Notable Activity_Had a Phone Conversation"] || 0);
    z_a += checkDummy("Last Notable Activity", lastNotableAct, "Modified") * (coef_a["Last Notable Activity_Modified"] || 0);
    z_a += checkDummy("Last Notable Activity", lastNotableAct, "SMS Sent") * (coef_a["Last Notable Activity_SMS Sent"] || 0);
    z_a += checkDummy("Last Notable Activity", lastNotableAct, "Unreachable") * (coef_a["Last Notable Activity_Unreachable"] || 0);
    
    const prob_a = 1.0 / (1.0 + Math.exp(-z_a));
    const score_a = Math.round(prob_a * 100);
    const cat_a = prob_a >= ModelParams.model_no_tags.optimal_cutoff ? "Hot Lead" : "Cold Lead";

    // 3. Score Model B (With Tags)
    let z_b = ModelParams.model_with_tags.intercept;
    const coef_b = ModelParams.model_with_tags.coefficients;
    
    z_b += scaled_time * (coef_b["Total Time Spent on Website"] || 0);
    z_b += checkDummy("Lead Origin", leadOrigin, "Landing Page Submission") * (coef_b["Lead Origin_Landing Page Submission"] || 0);
    z_b += checkDummy("Lead Source", leadSource, "Welingak Website") * (coef_b["Lead Source_Welingak Website"] || 0);
    z_b += checkDummy("Last Activity", lastActivity, "SMS Sent") * (coef_b["Last Activity_SMS Sent"] || 0);
    z_b += checkDummy("Tags", tags, "Busy") * (coef_b["Tags_Busy"] || 0);
    z_b += checkDummy("Tags", tags, "Closed by Horizzon") * (coef_b["Tags_Closed by Horizzon"] || 0);
    z_b += checkDummy("Tags", tags, "Lost to EINS") * (coef_b["Tags_Lost to EINS"] || 0);
    z_b += checkDummy("Tags", tags, "Not Specified") * (coef_b["Tags_Not Specified"] || 0);
    z_b += checkDummy("Tags", tags, "Ringing") * (coef_b["Tags_Ringing"] || 0);
    z_b += checkDummy("Tags", tags, "Want to take admission but has financial problems") * (coef_b["Tags_Want to take admission but has financial problems"] || 0);
    z_b += checkDummy("Tags", tags, "Will revert after reading the email") * (coef_b["Tags_Will revert after reading the email"] || 0);
    z_b += checkDummy("Tags", tags, "in touch with EINS") * (coef_b["Tags_in touch with EINS"] || 0);
    z_b += checkDummy("Tags", tags, "switched off") * (coef_b["Tags_switched off"] || 0);
    z_b += checkDummy("Last Notable Activity", lastNotableAct, "Email Link Clicked") * (coef_b["Last Notable Activity_Email Link Clicked"] || 0);
    z_b += checkDummy("Last Notable Activity", lastNotableAct, "Modified") * (coef_b["Last Notable Activity_Modified"] || 0);
    z_b += checkDummy("Last Notable Activity", lastNotableAct, "Olark Chat Conversation") * (coef_b["Last Notable Activity_Olark Chat Conversation"] || 0);
    
    const prob_b = 1.0 / (1.0 + Math.exp(-z_b));
    const score_b = Math.round(prob_b * 100);
    const cat_b = prob_b >= ModelParams.model_with_tags.optimal_cutoff ? "Hot Lead" : "Cold Lead";

    return {
        ...lead,
        score_no_tags: score_a,
        prob_no_tags: prob_a,
        cat_no_tags: cat_a,
        score_with_tags: score_b,
        prob_with_tags: prob_b,
        cat_with_tags: cat_b
    };
}

// Handle UI tab switching
function setTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    currentPage = 1;
    applyFiltersAndRender();
}

// Batch processing lead datasets
function processLeads(data) {
    leadsData = data.map(lead => scoreLead(lead));
    populateSimulatorDropdowns();
    applyFiltersAndRender();
}

// Calculate and render KPIs and charts
function applyFiltersAndRender() {
    const searchQuery = document.getElementById('search-box').value.toLowerCase();
    const sourceFilter = document.getElementById('filter-source').value;
    const priorityFilter = document.getElementById('filter-priority').value;

    filteredLeads = leadsData.filter(lead => {
        const matchesSearch = lead["Lead Number"].toString().includes(searchQuery) ||
                            (lead["Lead Source"] || "").toLowerCase().includes(searchQuery) ||
                            (lead["What is your current occupation"] || "").toLowerCase().includes(searchQuery);
        
        const matchesSource = !sourceFilter || lead["Lead Source"] === sourceFilter;
        
        let leadCategory = activeTab === 'no_tags' ? lead.cat_no_tags : lead.cat_with_tags;
        const matchesPriority = !priorityFilter || leadCategory === priorityFilter;
        
        return matchesSearch && matchesSource && matchesPriority;
    });

    renderKPIs();
    renderTable();
    renderCharts();
}

function renderKPIs() {
    const total = filteredLeads.length;
    document.getElementById('kpi-total-leads').innerText = total;

    if (total === 0) {
        document.getElementById('kpi-avg-score').innerText = '0';
        document.getElementById('kpi-hot-leads').innerText = '0';
        document.getElementById('kpi-projected-conv').innerText = '0%';
        return;
    }

    const scores = filteredLeads.map(l => activeTab === 'no_tags' ? l.score_no_tags : l.score_with_tags);
    const avgScore = Math.round(scores.reduce((a, b) => a + b, 0) / total);
    document.getElementById('kpi-avg-score').innerText = avgScore;

    const hotCount = filteredLeads.filter(l => activeTab === 'no_tags' ? l.cat_no_tags === 'Hot Lead' : l.cat_with_tags === 'Hot Lead').length;
    document.getElementById('kpi-hot-leads').innerText = hotCount;

    // Projected conversion rate: Out of the Hot Leads selected, what % actually convert in the dataset (if 'Converted' is present)
    const hasConvertedCol = filteredLeads.some(l => 'Converted' in l);
    if (hasConvertedCol) {
        const hotLeads = filteredLeads.filter(l => activeTab === 'no_tags' ? l.cat_no_tags === 'Hot Lead' : l.cat_with_tags === 'Hot Lead');
        if (hotLeads.length > 0) {
            const convertedHot = hotLeads.filter(l => parseInt(l.Converted) === 1).length;
            const projectedConv = Math.round((convertedHot / hotLeads.length) * 100);
            document.getElementById('kpi-projected-conv').innerText = `${projectedConv}%`;
        } else {
            document.getElementById('kpi-projected-conv').innerText = '0%';
        }
    } else {
        // Default target if Converted column not present
        document.getElementById('kpi-projected-conv').innerText = activeTab === 'no_tags' ? '77%' : '89%';
    }
}

function renderTable() {
    const tbody = document.querySelector('#leads-table tbody');
    tbody.innerHTML = '';

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = startIdx + pageSize;
    const pageLeads = filteredLeads.slice(startIdx, endIdx);

    if (pageLeads.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No leads found. Upload a CSV or change filters.</td></tr>`;
        updatePagination(0);
        return;
    }

    pageLeads.forEach(lead => {
        const score = activeTab === 'no_tags' ? lead.score_no_tags : lead.score_with_tags;
        const category = activeTab === 'no_tags' ? lead.cat_no_tags : lead.cat_with_tags;
        
        let scoreClass = 'score-low';
        let badgeClass = 'badge-cold';
        if (category === 'Hot Lead') {
            scoreClass = 'score-high';
            badgeClass = 'badge-hot';
        } else if (score >= 20) { // arbitrary warning boundary for warm
            scoreClass = 'score-medium';
            badgeClass = 'badge-warm';
        }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>#${lead["Lead Number"]}</strong></td>
            <td>${lead["Lead Source"] || 'Google'}</td>
            <td>${lead["What is your current occupation"] || 'Unemployed'}</td>
            <td>${lead["Last Activity"] || 'Email Opened'}</td>
            <td>${activeTab === 'with_tags' ? (lead["Tags"] || 'Not Specified') : 'N/A (Excluded)'}</td>
            <td><span class="score-text ${scoreClass}">${score}</span></td>
            <td><span class="badge ${badgeClass}">${category}</span></td>
        `;
        tbody.appendChild(tr);
    });

    updatePagination(filteredLeads.length);
}

function updatePagination(totalCount) {
    const totalPages = Math.ceil(totalCount / pageSize) || 1;
    document.getElementById('page-info').innerText = `Page ${currentPage} of ${totalPages} (${totalCount} total leads)`;
    document.getElementById('btn-prev').disabled = currentPage === 1;
    document.getElementById('btn-next').disabled = currentPage === totalPages;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        renderTable();
    }
}

function nextPage() {
    const totalPages = Math.ceil(filteredLeads.length / pageSize) || 1;
    if (currentPage < totalPages) {
        currentPage++;
        renderTable();
    }
}

// Populate drop-down filter list based on active sources
function populateSourceFilter() {
    const sources = [...new Set(leadsData.map(l => l["Lead Source"]).filter(Boolean))];
    const filter = document.getElementById('filter-source');
    filter.innerHTML = '<option value="">All Sources</option>';
    sources.sort().forEach(src => {
        filter.innerHTML += `<option value="${src}">${src}</option>`;
    });
}

// Populate simulator drop-downs based on active data
function populateSimulatorDropdowns() {
    const fields = [
        { id: 'sim-email', key: 'Do Not Email', defaultVal: 'No' },
        { id: 'sim-origin', key: 'Lead Origin', defaultVal: 'Landing Page Submission' },
        { id: 'sim-source', key: 'Lead Source', defaultVal: 'Google' },
        { id: 'sim-occ', key: 'What is your current occupation', defaultVal: 'Unemployed' },
        { id: 'sim-spec', key: 'Specialization', defaultVal: 'Not Specified' },
        { id: 'sim-activity', key: 'Last Activity', defaultVal: 'Email Opened' },
        { id: 'sim-notable', key: 'Last Notable Activity', defaultVal: 'Modified' },
        { id: 'sim-tag', key: 'Tags', defaultVal: 'Not Specified' }
    ];

    fields.forEach(field => {
        // Collect all unique non-empty values for this field from leadsData
        const uniqueValues = [...new Set(leadsData.map(l => l[field.key]).filter(Boolean))].sort();
        const selectEl = document.getElementById(field.id);
        
        if (selectEl) {
            selectEl.innerHTML = '';
            
            // Add the unique values as options
            uniqueValues.forEach(val => {
                const opt = document.createElement('option');
                opt.value = val;
                opt.textContent = val;
                if (val === field.defaultVal) {
                    opt.selected = true;
                }
                selectEl.appendChild(opt);
            });
            
            // Fallback: If default value isn't in data, add it so form has a valid option
            if (!uniqueValues.includes(field.defaultVal)) {
                const opt = document.createElement('option');
                opt.value = field.defaultVal;
                opt.textContent = field.defaultVal;
                opt.selected = true;
                selectEl.appendChild(opt);
            }
        }
    });
}

// Render graphical reports using Chart.js
function renderCharts() {
    const scoreData = { hot: 0, warm: 0, cold: 0 };
    filteredLeads.forEach(l => {
        const score = activeTab === 'no_tags' ? l.score_no_tags : l.score_with_tags;
        const category = activeTab === 'no_tags' ? l.cat_no_tags : l.cat_with_tags;
        if (category === 'Hot Lead') scoreData.hot++;
        else if (score >= 15) scoreData.warm++;
        else scoreData.cold++;
    });

    // Score distribution chart (Pie/Doughnut)
    if (scoreChart) scoreChart.destroy();
    const ctxScore = document.getElementById('score-distribution-chart').getContext('2d');
    scoreChart = new Chart(ctxScore, {
        type: 'doughnut',
        data: {
            labels: ['Hot Leads (High Probability)', 'Warm Leads (Medium Interest)', 'Cold Leads (Low Interest)'],
            datasets: [{
                data: [scoreData.hot, scoreData.warm, scoreData.cold],
                backgroundColor: ['#000000', '#777777', '#CCCCCC'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#000000', font: { family: 'Inter', size: 11, weight: 'bold' } }
                }
            }
        }
    });

    // Conversion rate by Lead Source chart (Bar Chart)
    const sourceStats = {};
    filteredLeads.forEach(l => {
        const src = l["Lead Source"] || 'Google';
        if (!sourceStats[src]) sourceStats[src] = { total: 0, hot: 0 };
        sourceStats[src].total++;
        const category = activeTab === 'no_tags' ? l.cat_no_tags : l.cat_with_tags;
        if (category === 'Hot Lead') sourceStats[src].hot++;
    });

    const sortedSources = Object.keys(sourceStats)
        .map(name => ({ name, ...sourceStats[name] }))
        .sort((a, b) => b.total - a.total)
        .slice(0, 5);

    if (sourceChart) sourceChart.destroy();
    const ctxSource = document.getElementById('lead-sources-chart').getContext('2d');
    sourceChart = new Chart(ctxSource, {
        type: 'bar',
        data: {
            labels: sortedSources.map(s => s.name),
            datasets: [
                {
                    label: 'Total Leads',
                    data: sortedSources.map(s => s.total),
                    backgroundColor: '#CCCCCC',
                    borderRadius: 0
                },
                {
                    label: 'Hot Leads',
                    data: sortedSources.map(s => s.hot),
                    backgroundColor: '#000000',
                    borderRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: '#000000', font: { weight: 'bold' } }, grid: { display: false } },
                y: { ticks: { color: '#000000', font: { weight: 'bold' } }, grid: { color: 'rgba(0,0,0,0.1)' } }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#000000', font: { family: 'Inter', size: 11, weight: 'bold' } }
                }
            }
        }
    });
}

// Simulator functionality
function runSimulator() {
    const doNotEmail = document.getElementById('sim-email').value;
    const timeSpent = parseFloat(document.getElementById('sim-time').value || 0);
    const origin = document.getElementById('sim-origin').value;
    const source = document.getElementById('sim-source').value;
    const specialization = document.getElementById('sim-spec').value;
    const occupation = document.getElementById('sim-occ').value;
    const lastActivity = document.getElementById('sim-activity').value;
    const lastNotableAct = document.getElementById('sim-notable').value;
    const tag = document.getElementById('sim-tag').value;

    const leadInput = {
        "Do Not Email": doNotEmail,
        "Total Time Spent on Website": timeSpent,
        "Lead Origin": origin,
        "Lead Source": source,
        "Specialization": specialization,
        "What is your current occupation": occupation,
        "Last Activity": lastActivity,
        "Last Notable Activity": lastNotableAct,
        "Tags": tag,
        "TotalVisits": 3,  // default values for optional items
        "Page Views Per Visit": 2.0
    };

    const result = scoreLead(leadInput);
    
    // Display result box
    const outBox = document.getElementById('sim-output');
    outBox.style.display = 'block';

    const simScore = activeTab === 'no_tags' ? result.score_no_tags : result.score_with_tags;
    const simProb = activeTab === 'no_tags' ? result.prob_no_tags : result.prob_with_tags;
    const simCat = activeTab === 'no_tags' ? result.cat_no_tags : result.cat_with_tags;

    document.getElementById('sim-out-score').innerText = simScore;
    document.getElementById('sim-out-score').className = 'output-score ' + (simCat === 'Hot Lead' ? 'score-high' : 'score-low');
    document.getElementById('sim-out-prob').innerText = `Conversion Likelihood: ${(simProb * 100).toFixed(2)}% (Optimal Threshold: ${activeTab === 'no_tags' ? '40%' : '30%'})`;
    
    const adviceDiv = document.getElementById('sim-out-advice');
    if (simCat === 'Hot Lead') {
        adviceDiv.className = 'output-advice advice-hot';
        adviceDiv.innerHTML = `<strong>Priority Action: Hot Lead!</strong><br>Prioritize this prospect immediately. Contact via phone within 1 hour. Target sales strategy on their background (${occupation}).`;
    } else {
        adviceDiv.className = 'output-advice advice-cold';
        adviceDiv.innerHTML = `<strong>Priority Action: Cold Lead (Nurturing)</strong><br>Do not spend sales call bandwidth on this prospect yet. Route to automated email nurturing sequences and monitor website activities for score changes.`;
    }
}

// Parse file uploads client side
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Use PapaParse to parse local file
    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function(results) {
            console.log("File loaded successfully, parsing rows:", results.data.length);
            if (results.data.length > 0) {
                processLeads(results.data);
                populateSourceFilter();
            } else {
                alert("The CSV file appears to be empty.");
            }
        },
        error: function(err) {
            alert("Error parsing CSV: " + err.message);
        }
    });
}

// Wire up events on window load
window.addEventListener('DOMContentLoaded', () => {
    // Load initial mock data
    processLeads(mockLeads);
    populateSourceFilter();

    // File drag and drop / upload handler
    const fileInput = document.getElementById('csv-file');
    const uploadZone = document.getElementById('upload-zone');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);

    // Filter events
    document.getElementById('search-box').addEventListener('input', applyFiltersAndRender);
    document.getElementById('filter-source').addEventListener('change', applyFiltersAndRender);
    document.getElementById('filter-priority').addEventListener('change', applyFiltersAndRender);

    // Tab switcher
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            setTab(e.target.dataset.tab);
        });
    });

    // Pagination
    document.getElementById('btn-prev').addEventListener('click', prevPage);
    document.getElementById('btn-next').addEventListener('click', nextPage);

    // Simulator Submit
    document.getElementById('btn-run-sim').addEventListener('click', runSimulator);
});
