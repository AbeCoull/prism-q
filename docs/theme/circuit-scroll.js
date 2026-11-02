for (const panel of document.querySelectorAll('.circuit-scroll, .mermaid')) {
    const isDiagram = panel.matches('.mermaid');
    const hint = document.createElement('p');
    hint.className = 'circuit-overflow';
    hint.hidden = true;
    if (isDiagram) {
        panel.before(hint);
    } else {
        panel.after(hint);
    }
    let diagramPositioned = false;

    const updateOverflow = () => {
        const svg = isDiagram ? panel.querySelector('svg') : null;
        const node = svg?.querySelector('.node');
        if (!diagramPositioned && svg?.viewBox.baseVal.width > 0 && node) {
            const nodeBounds = node.getBoundingClientRect();
            if (nodeBounds.width > 0) {
                const viewport = panel.getBoundingClientRect();
                panel.scrollLeft += nodeBounds.x + nodeBounds.width / 2
                    - viewport.x - panel.clientLeft - panel.clientWidth / 2;
                diagramPositioned = true;
            }
        }
        const moreLeft = panel.scrollLeft > 1;
        const moreRight = panel.scrollWidth - panel.clientWidth - panel.scrollLeft > 1;
        hint.hidden = !moreLeft && !moreRight;
        hint.textContent = moreLeft && moreRight
            ? 'Scroll left or right for more of this figure.'
            : moreLeft
                ? 'Scroll left for more of this figure.'
                : 'Scroll right for more of this figure.';
    };

    panel.addEventListener('keydown', event => {
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
            event.stopPropagation();
        }
    });
    panel.addEventListener('scroll', updateOverflow, { passive: true });
    for (const image of panel.querySelectorAll('img')) {
        image.addEventListener('load', updateOverflow);
    }
    new ResizeObserver(updateOverflow).observe(panel);
    updateOverflow();
}
