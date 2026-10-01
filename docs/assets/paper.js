"use strict";
const clientData = JSON.parse(document.getElementById("client-data").textContent);
const select = document.getElementById("client-select");
const takeaway = document.getElementById("client-takeaway");
select.addEventListener("change", () => {
  const selected = select.value;
  document.querySelectorAll("#client-results tbody tr").forEach(row => {
    row.hidden = selected !== "all" && row.dataset.client !== selected;
  });
  if (selected === "all") {
    takeaway.textContent = "The model leads for four clients; the rule leads for three. Each client has equal weight in the overall Precision@20.";
    return;
  }
  const client = clientData.find(item => item.client_alias === selected);
  const gap = client.model_minus_rule_percentage_points;
  const leader = gap > 0 ? "the model" : "the rule";
  takeaway.textContent = `${client.client_alias}: ${leader} leads by ${Math.abs(gap).toFixed(1)} percentage points on 20 selections, among ${client.pages.toLocaleString("en-US")} eligible pages. This is a descriptive result, not a client-specific deployment policy.`;
});
select.dispatchEvent(new Event("change"));

const sections = document.querySelectorAll("main section[id]");
const navLinks = document.querySelectorAll(".sidebar nav a");
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      navLinks.forEach(link => {
        if (link.getAttribute("href") === `#${entry.target.id}`) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    }
  }, {rootMargin: "-15% 0px -65% 0px", threshold: 0});
  sections.forEach(section => observer.observe(section));
}

let closedBeforePrint = [];
let selectedBeforePrint = "all";
window.addEventListener("beforeprint", () => {
  selectedBeforePrint = select.value;
  select.value = "all";
  select.dispatchEvent(new Event("change"));
  closedBeforePrint = [...document.querySelectorAll("details:not([open])")];
  closedBeforePrint.forEach(details => details.open = true);
});
window.addEventListener("afterprint", () => {
  closedBeforePrint.forEach(details => details.open = false);
  select.value = selectedBeforePrint;
  select.dispatchEvent(new Event("change"));
});
document.getElementById("print-paper").addEventListener("click", () => window.print());
