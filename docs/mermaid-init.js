// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

(() => {
    const diagrams = document.querySelectorAll('.mermaid');
    if (!diagrams.length) {
        return;
    }

    for (const diagram of diagrams) {
        diagram.setAttribute('tabindex', '0');
        diagram.setAttribute('role', 'region');
        diagram.setAttribute('aria-label', 'Diagram');
    }

    const darkThemes = ['ayu', 'navy', 'coal'];
    const root = document.documentElement;
    const style = getComputedStyle(root);
    const color = name => style.getPropertyValue(name).trim();
    const surface = color('--doc-code-bg');
    const text = color('--fg');
    const border = color('--doc-line-strong');

    mermaid.initialize({
        startOnLoad: true,
        theme: 'base',
        themeVariables: {
            darkMode: darkThemes.some(theme => root.classList.contains(theme)),
            background: color('--bg'),
            fontFamily: style.fontFamily,
            fontSize: '14px',
            primaryColor: surface,
            primaryTextColor: text,
            primaryBorderColor: border,
            secondaryColor: surface,
            secondaryTextColor: text,
            secondaryBorderColor: border,
            tertiaryColor: surface,
            tertiaryTextColor: text,
            tertiaryBorderColor: border,
            lineColor: color('--doc-muted'),
            textColor: text,
            edgeLabelBackground: color('--bg'),
            noteBkgColor: surface,
            noteTextColor: text,
            noteBorderColor: border,
        },
        flowchart: { curve: 'linear', useMaxWidth: false },
    });

    for (const button of document.querySelectorAll('#theme-list .theme')) {
        button.addEventListener('click', () => window.location.reload());
    }
})();
