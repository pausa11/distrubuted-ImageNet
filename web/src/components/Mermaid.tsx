"use client";
import React, { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

mermaid.initialize({
    startOnLoad: false,
    theme: "default",
    securityLevel: "loose",
});

interface MermaidProps {
    chart: string;
}

export default function Mermaid({ chart }: MermaidProps) {
    const ref = useRef<HTMLDivElement>(null);
    const [svg, setSvg] = useState<string>("");

    useEffect(() => {
        if (chart) {
            const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
            mermaid.render(id, chart).then(({ svg }) => {
                setSvg(svg);
            });
        }
    }, [chart]);

    return (
        <div
            ref={ref}
            className="mermaid overflow-x-auto flex justify-center p-4 bg-white rounded-lg border border-gray-100"
            dangerouslySetInnerHTML={{ __html: svg }}
        />
    );
}
