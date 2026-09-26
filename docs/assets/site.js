"use strict";

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
function activateTab(tab) {
  for (const item of tabs) {
    const active = item === tab;
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden = !active;
  }
}
for (const tab of tabs) {
  tab.addEventListener("click", () => activateTab(tab));
  tab.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(tab);
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    activateTab(tabs[next]);
    tabs[next].focus();
  });
}

const copyButton = document.getElementById("copy-command");
copyButton.addEventListener("click", async () => {
  const code = document.querySelector('[role="tabpanel"]:not([hidden]) pre code');
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(code.textContent);
    copyButton.textContent = "Copied ✓";
    status.textContent = "Commands copied to clipboard.";
  } catch {
    copyButton.textContent = "Select & copy";
    status.textContent = "Clipboard access unavailable. Commands selected for manual copying.";
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
  }
  setTimeout(() => { copyButton.textContent = "Copy commands"; }, 2200);
});
